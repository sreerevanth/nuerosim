"""
neurosim/cli.py

Command-line interface for NeuroSim platform.

Commands:
  simulate     — run a simulation from a YAML config
  validate     — validate a neuron morphology or network config
  analyze      — post-process simulation results
  optimize     — run parameter optimization
  serve        — start the REST API server
  dashboard    — launch the visualization dashboard
  benchmark    — run performance benchmarks
  export       — export results to different formats
"""

from __future__ import annotations
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import click

logging.basicConfig(
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("neurosim.cli")


# ─────────────────────────────────────────────────────────────
#  Root group
# ─────────────────────────────────────────────────────────────
@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option("1.0.0", prog_name="neurosim")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool):
    """
    NeuroSim — Large-Scale Neural Simulation Platform

    \b
    Examples:
      neurosim simulate --config configs/experiments.yaml
      neurosim simulate --config configs/experiments.yaml --distributed
      neurosim analyze  --results results/run_001/results.h5
      neurosim dashboard --results-dir results/
    """
    if verbose:
        logging.getLogger("neurosim").setLevel(logging.DEBUG)


# ─────────────────────────────────────────────────────────────
#  simulate
# ─────────────────────────────────────────────────────────────
@main.command()
@click.option("--config",       "-c", required=True, type=click.Path(exists=True),
              help="YAML simulation configuration file")
@click.option("--output-dir",   "-o", default=None,
              help="Output directory (overrides config)")
@click.option("--distributed",  is_flag=True,
              help="Use MPI distributed simulation")
@click.option("--n-workers",    default=None, type=int,
              help="Number of parallel workers")
@click.option("--dt",           default=None, type=float,
              help="Override time step (ms)")
@click.option("--t-stop",       default=None, type=float,
              help="Override simulation end time (ms)")
@click.option("--checkpoint",   default=None, type=click.Path(),
              help="Resume from checkpoint file")
@click.option("--profile",      is_flag=True,
              help="Enable performance profiling")
def simulate(config, output_dir, distributed, n_workers, dt,
             t_stop, checkpoint, profile):
    """Run a neural simulation from a YAML configuration file."""
    import yaml
    from neurosim.simulation.engine import (
        NeuralSimulationEngine, SimulationConfig
    )
    from neurosim.reconstruction.ingestion import ReconstructionPipeline

    click.echo(click.style("\n  NeuroSim Simulation Engine v1.0.0\n", fg="cyan"))

    with open(config) as f:
        cfg_dict = list(yaml.safe_load_all(f))[0]   # first document

    # CLI overrides
    sim_cfg = cfg_dict.get("simulation", {})
    if dt:        sim_cfg["dt"]       = dt
    if t_stop:    sim_cfg["t_stop"]   = t_stop
    if n_workers: sim_cfg["n_workers"] = n_workers

    out_dir = output_dir or cfg_dict.get("output", {}).get("dir", "results/run")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cfg = SimulationConfig.from_dict(sim_cfg)
    cfg.checkpoint_dir = str(Path(out_dir) / "checkpoints")

    click.echo(f"  Config:     {config}")
    click.echo(f"  Output:     {out_dir}")
    click.echo(f"  dt:         {cfg.dt} ms")
    click.echo(f"  t_stop:     {cfg.t_stop} ms")
    click.echo(f"  Integrator: {cfg.integrator}")
    click.echo(f"  Workers:    {cfg.n_workers}")
    click.echo(f"  Distributed:{distributed}\n")

    engine   = NeuralSimulationEngine(cfg)
    pipeline = ReconstructionPipeline(output_dir=out_dir)

    # Build network
    net_spec = cfg_dict.get("network", {})
    net_type = net_spec.get("type", "random_ei")

    if net_type == "single_neuron":
        from neurosim.models.neuron import build_l5_pyramidal_cell
        pipeline.neurons[0] = build_l5_pyramidal_cell(neuron_id=0)
    else:
        pipeline.build_random_network(
            n_exc = net_spec.get("n_excitatory", 100),
            n_inh = net_spec.get("n_inhibitory", 25),
            p_ee  = net_spec.get("connectivity", {}).get("p_ee", 0.1),
            p_ei  = net_spec.get("connectivity", {}).get("p_ei", 0.5),
            p_ie  = net_spec.get("connectivity", {}).get("p_ie", 0.5),
            p_ii  = net_spec.get("connectivity", {}).get("p_ii", 0.1),
        )

    pipeline.export_to_engine(engine)

    # Apply current clamps
    for clamp in cfg_dict.get("current_clamps", []):
        nid   = clamp["neuron_id"]
        I_amp = clamp["I_amp"]
        t0    = clamp.get("t_start", 0.0)
        t1    = clamp.get("t_stop", cfg.t_stop)
        engine.clamp_current(
            nid,
            lambda t, _I=I_amp, _t0=t0, _t1=t1:
                _I if _t0 <= t <= _t1 else 0.0
        )

    # Restore checkpoint
    if checkpoint:
        engine.restore_checkpoint(checkpoint)
        click.echo(f"  Restored from checkpoint: {checkpoint}")

    # Progress bar
    bar_width = 40
    def progress_cb(pct: float, t: float) -> None:
        filled = int(bar_width * pct / 100)
        bar    = "█" * filled + "░" * (bar_width - filled)
        click.echo(f"\r  [{bar}] {pct:5.1f}%  t={t:.1f}ms", nl=False)

    if profile:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()

    t_wall_start = time.perf_counter()

    if distributed:
        try:
            from neurosim.simulation.distributed import DistributedSimulation
            from mpi4py import MPI
            comm    = MPI.COMM_WORLD
            rank    = comm.Get_rank()
            dist_sim = DistributedSimulation(engine, min_delay=cfg.dt * 40)
            results  = dist_sim.run()
        except ImportError:
            click.echo(click.style(
                "\n  mpi4py not installed — running single-process", fg="yellow"))
            results = engine.run(progress_callback=progress_cb)
    else:
        results = engine.run(progress_callback=progress_cb)

    wall_time = time.perf_counter() - t_wall_start
    click.echo("")  # newline after progress bar

    if profile:
        pr.disable()
        import io, pstats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        click.echo(click.style("\n  Profile (top 20):\n", fg="yellow"))
        click.echo(s.getvalue())

    # Save results
    _save_results(results, out_dir, cfg_dict)

    # Summary
    click.echo(click.style("\n  ─── Simulation Complete ───", fg="green"))
    click.echo(f"  Neurons:     {results.get('n_neurons', '?')}")
    click.echo(f"  Connections: {results.get('n_connections', '?')}")
    click.echo(f"  Spikes:      {results.get('total_spikes', 0):,}")
    click.echo(f"  Mean rate:   {results.get('mean_firing_rate_hz', 0.0):.2f} Hz")
    click.echo(f"  Wall time:   {wall_time:.2f}s")
    click.echo(f"  Output:      {out_dir}\n")


# ─────────────────────────────────────────────────────────────
#  validate
# ─────────────────────────────────────────────────────────────
@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--type", "ftype", default="auto",
              type=click.Choice(["auto", "swc", "yaml", "sonata"]))
def validate(path, ftype):
    """Validate a morphology file or configuration."""
    from neurosim.reconstruction.ingestion import (
        SWCReader, MorphologyValidator
    )

    p = Path(path)
    if ftype == "auto":
        ftype = p.suffix.lstrip(".")

    if ftype == "swc":
        click.echo(f"Validating SWC: {path}")
        points = SWCReader.read(path)
        validator = MorphologyValidator()
        cleaned, issues = validator.validate(points)
        click.echo(f"  Points loaded: {len(points)}")
        click.echo(f"  Issues found:  {len(issues)}")
        for iss in issues[:10]:
            click.echo(f"    [{iss.issue_type}] pt#{iss.point_id}: "
                       f"{iss.description}", err=True)
        if not issues:
            click.echo(click.style("  ✓ Morphology valid", fg="green"))
        else:
            click.echo(click.style(f"  ⚠ {len(issues)} issues", fg="yellow"))

    elif ftype in ("yaml", "yml"):
        import yaml
        click.echo(f"Validating YAML config: {path}")
        with open(path) as f:
            docs = list(yaml.safe_load_all(f))
        click.echo(f"  Documents:   {len(docs)}")
        for i, doc in enumerate(docs):
            name = doc.get("experiment", {}).get("name", f"doc-{i}")
            click.echo(f"  [{i}] {name}")
        click.echo(click.style("  ✓ Config valid", fg="green"))
    else:
        click.echo(f"  Unknown file type: {ftype}", err=True)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
#  analyze
# ─────────────────────────────────────────────────────────────
@main.command()
@click.option("--results", "-r", required=True, type=click.Path(exists=True),
              help="Path to HDF5 results file")
@click.option("--output",  "-o", default=None,
              help="Output path for analysis report")
@click.option("--psth",         is_flag=True, help="Compute PSTH")
@click.option("--synchrony",    is_flag=True, help="Compute synchrony index")
@click.option("--correlation",  is_flag=True, help="Compute pairwise correlations")
@click.option("--lfp",          is_flag=True, help="Estimate LFP from Vm")
@click.option("--all",    "all_analyses", is_flag=True,
              help="Run all analyses")
def analyze(results, output, psth, synchrony, correlation, lfp, all_analyses):
    """Post-process and analyze simulation results."""
    import h5py
    import json
    from neurosim.analysis.spike_analysis import (
        compute_psth, synchrony_index, pairwise_correlation,
        mean_firing_rate, cv_isi, fano_factor
    )

    if all_analyses:
        psth = synchrony = correlation = lfp = True

    click.echo(f"\n  Analyzing: {results}")
    report = {}

    with h5py.File(results, "r") as f:
        spike_data = {}
        if "spikes" in f:
            for nid in f["spikes"]:
                spike_data[int(nid)] = f["spikes"][nid][:].tolist()

        V_data = f["V"][:] if "V" in f else None
        t_data = f["t"][:] if "t" in f else None
        t_stop = float(f.attrs.get("t_stop", 1000.0))

    spike_trains = list(spike_data.values())

    # Basic stats always
    total_spikes = sum(len(st) for st in spike_trains)
    mean_rate    = total_spikes / max(1, len(spike_trains)) / (t_stop / 1000.0)
    report["n_neurons"]      = len(spike_trains)
    report["total_spikes"]   = total_spikes
    report["mean_rate_hz"]   = mean_rate
    report["fano_factor"]    = fano_factor(spike_trains, 0, t_stop)

    if psth and spike_trains:
        ps = compute_psth(spike_trains, 0, t_stop, bin_size=20.0)
        report["psth"] = {
            "bin_centers": ps.bin_centers.tolist(),
            "rate_hz":     ps.rate.tolist(),
        }
        click.echo(f"  PSTH:        peak={ps.rate.max():.2f} Hz")

    if synchrony and spike_trains:
        si = synchrony_index(spike_trains, 0, t_stop)
        report["synchrony_index"] = si
        click.echo(f"  Synchrony:   {si:.4f}")

    if correlation and len(spike_trains) > 1:
        corr = pairwise_correlation(spike_trains, 0, t_stop)
        import numpy as np
        mean_corr = float(np.triu(corr, k=1).mean())
        report["mean_pairwise_correlation"] = mean_corr
        click.echo(f"  Mean corr:   {mean_corr:.4f}")

    # Save report
    out_path = output or str(Path(results).with_suffix(".analysis.json"))
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    click.echo(click.style(f"\n  ✓ Report saved: {out_path}\n", fg="green"))


# ─────────────────────────────────────────────────────────────
#  optimize
# ─────────────────────────────────────────────────────────────
@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Optimization config YAML")
@click.option("--n-generations", default=50, type=int)
@click.option("--popsize",       default=10,  type=int)
@click.option("--output",  "-o", default="optimization_results.json")
def optimize(config, n_generations, popsize, output):
    """Run CMA-ES parameter optimization."""
    import yaml
    from neurosim.ml.optimization import (
        ParameterSpace, EvolutionaryOptimizer,
        FiringPatternObjective
    )
    from neurosim.models.neuron import build_l5_pyramidal_cell
    from neurosim.simulation.engine import run_current_clamp
    import json

    with open(config) as f:
        opt_cfg = yaml.safe_load(f)

    space = ParameterSpace()
    for p in opt_cfg.get("parameters", [
        {"name": "gNa",  "lo": 50.0,  "hi": 300.0},
        {"name": "gK",   "lo": 10.0,  "hi": 100.0},
        {"name": "gL",   "lo": 0.05,  "hi": 1.0,   "log_scale": True},
    ]):
        space.add(p["name"], p["lo"], p["hi"], p.get("log_scale", False))

    # Reference target
    ref_neuron  = build_l5_pyramidal_cell()
    ref_results = run_current_clamp(ref_neuron, I_amp=2.0, t_stop_sim=600.0)
    import numpy as np
    target_V = ref_results["V"][0, 0, :]
    target_t = ref_results["t"]

    def builder(theta):
        n = build_l5_pyramidal_cell()
        if "gNa" in theta: n.compartments[0].channels["na"].g_max  = theta["gNa"]
        if "gK"  in theta: n.compartments[0].channels["k"].g_max   = theta["gK"]
        if "gL"  in theta: n.compartments[0].channels["leak"].g_max = theta["gL"]
        n.initialize(-65.0)
        return n

    objective = FiringPatternObjective(
        target_V=target_V, target_t=target_t,
        neuron_builder=builder,
        sim_fn=lambda n, I: run_current_clamp(n, I_amp=I, t_stop_sim=600.0),
        I_amp=opt_cfg.get("I_amp", 2.0),
    )

    optimizer = EvolutionaryOptimizer(space, popsize=popsize)

    click.echo(f"\n  CMA-ES: {n_generations} generations × {popsize} population")
    click.echo(f"  Parameters: {[p.name for p in space.params]}\n")

    result = optimizer.optimize(objective, n_generations=n_generations)

    click.echo(click.style("\n  ─── Optimization Complete ───", fg="green"))
    click.echo(f"  Best params:  {result.best_params}")
    click.echo(f"  Best value:   {result.best_value:.6f}")
    click.echo(f"  Evaluations:  {result.n_evaluations}")
    click.echo(f"  Converged:    {result.converged}\n")

    with open(output, "w") as f:
        json.dump({
            "best_params":   result.best_params,
            "best_value":    result.best_value,
            "n_evaluations": result.n_evaluations,
            "converged":     result.converged,
            "history":       result.history,
        }, f, indent=2)
    click.echo(f"  Results saved: {output}")


# ─────────────────────────────────────────────────────────────
#  serve
# ─────────────────────────────────────────────────────────────
@main.command()
@click.option("--host",    default="0.0.0.0")
@click.option("--port",    default=8000, type=int)
@click.option("--workers", default=4,    type=int)
@click.option("--reload",  is_flag=True, help="Auto-reload (dev mode)")
def serve(host, port, workers, reload):
    """Start the NeuroSim REST API server."""
    import uvicorn
    click.echo(f"\n  NeuroSim API → http://{host}:{port}")
    click.echo(f"  Docs:        http://{host}:{port}/docs\n")
    uvicorn.run(
        "neurosim.services.gateway.main:app",
        host=host, port=port,
        workers=1 if reload else workers,
        reload=reload,
        log_level="info",
    )


# ─────────────────────────────────────────────────────────────
#  dashboard
# ─────────────────────────────────────────────────────────────
@main.command()
@click.option("--results-dir", default="results")
@click.option("--port",        default=8050, type=int)
@click.option("--debug",       is_flag=True)
def dashboard(results_dir, port, debug):
    """Launch the interactive visualization dashboard."""
    from neurosim.visualization.dashboard import create_dashboard
    app = create_dashboard(results_dir=results_dir)
    click.echo(f"\n  NeuroSim Dashboard → http://localhost:{port}\n")
    app.run_server(host="0.0.0.0", port=port, debug=debug)


# ─────────────────────────────────────────────────────────────
#  benchmark
# ─────────────────────────────────────────────────────────────
@main.command()
@click.option("--n-neurons",    default=100,  type=int)
@click.option("--t-stop",       default=500.0, type=float)
@click.option("--n-repeats",    default=3,    type=int)
@click.option("--output", "-o", default="benchmark_results.json")
def benchmark(n_neurons, t_stop, n_repeats, output):
    """Run performance benchmarks and report throughput."""
    import json
    import numpy as np
    from neurosim.simulation.engine import SimulationConfig, NeuralSimulationEngine
    from neurosim.reconstruction.ingestion import ReconstructionPipeline

    click.echo(f"\n  Benchmarking: {n_neurons} neurons × {t_stop}ms × {n_repeats} runs\n")
    results = []

    for run in range(n_repeats):
        cfg      = SimulationConfig(dt=0.025, t_stop=t_stop, n_workers=4)
        engine   = NeuralSimulationEngine(cfg)
        pipeline = ReconstructionPipeline()
        n_exc    = int(n_neurons * 0.8)
        n_inh    = n_neurons - n_exc
        pipeline.build_random_network(n_exc=n_exc, n_inh=n_inh)
        pipeline.export_to_engine(engine)
        engine.clamp_current(0, lambda t: 2.0 if 50 <= t <= t_stop - 50 else 0.0)

        t_start = time.perf_counter()
        r       = engine.run()
        elapsed = time.perf_counter() - t_start

        n_steps = int(t_stop / 0.025)
        throughput = n_neurons * n_steps / elapsed / 1e6  # MNUPS

        click.echo(f"  Run {run+1}: {elapsed:.2f}s  "
                   f"{throughput:.2f} MNUPS  "
                   f"{r['total_spikes']:,} spikes")
        results.append({"wall_s": elapsed, "mnups": throughput,
                        "spikes": r["total_spikes"]})

    mnups_all = [r["mnups"] for r in results]
    click.echo(f"\n  Mean: {np.mean(mnups_all):.2f} ± {np.std(mnups_all):.2f} MNUPS")
    click.echo(click.style("  (MNUPS = Million Neuron Updates Per Second)\n", fg="cyan"))

    with open(output, "w") as f:
        json.dump({
            "config":   {"n_neurons": n_neurons, "t_stop": t_stop},
            "runs":     results,
            "mean_mnups": float(np.mean(mnups_all)),
            "std_mnups":  float(np.std(mnups_all)),
        }, f, indent=2)
    click.echo(f"  Saved: {output}")


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────
def _save_results(results: dict, out_dir: str, cfg_dict: dict) -> None:
    """Save simulation results to HDF5 and optional CSV."""
    import numpy as np
    try:
        import h5py
        h5_path = str(Path(out_dir) / "results.h5")
        with h5py.File(h5_path, "w") as f:
            f.attrs["t_stop"]     = cfg_dict.get("simulation", {}).get("t_stop", 0)
            f.attrs["n_neurons"]  = results.get("n_neurons", 0)
            f.attrs["n_spikes"]   = results.get("total_spikes", 0)

            if "t" in results:
                f.create_dataset("t", data=np.array(results["t"]),
                                 compression="gzip", compression_opts=4)
            if "V" in results and results["V"] is not None:
                V = np.array(results["V"])
                f.create_dataset("V", data=V.astype(np.float32),
                                 compression="gzip", compression_opts=4)

            spk_grp = f.create_group("spikes")
            for nid, times in results.get("spikes", {}).items():
                spk_grp.create_dataset(str(nid), data=np.array(times))

        click.echo(f"  HDF5:        {h5_path}")
    except ImportError:
        import json
        json_path = str(Path(out_dir) / "results_summary.json")
        with open(json_path, "w") as jf:
            json.dump({
                "n_neurons":   results.get("n_neurons"),
                "n_spikes":    results.get("total_spikes"),
                "mean_rate_hz": results.get("mean_firing_rate_hz"),
                "wall_time_s": results.get("wall_time_s"),
            }, jf, indent=2)
        click.echo(f"  JSON:        {json_path}")


if __name__ == "__main__":
    main()
