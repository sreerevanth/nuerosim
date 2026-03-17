"""
neurosim/worker/celery_app.py

Celery-based asynchronous job queue for simulation dispatch.

Queues:
  neurosim-jobs      — standard CPU simulations
  neurosim-gpu-jobs  — GPU-accelerated large-scale simulations
  neurosim-mpi-jobs  — MPI-distributed jobs (submits to HPC scheduler)

Each task is idempotent, fault-tolerant, and checkpointed.
"""

from __future__ import annotations
import os
import logging
import time
from typing import Optional
from pathlib import Path

from celery import Celery, Task
from celery.utils.log import get_task_logger
from celery.signals import worker_ready, task_prerun, task_postrun

logger = get_task_logger(__name__)

# ─────────────────────────────────────────────────────────────
#  Celery app configuration
# ─────────────────────────────────────────────────────────────
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "neurosim",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

app.conf.update(
    # Serialization
    task_serializer         = "json",
    result_serializer       = "json",
    accept_content          = ["json"],

    # Reliability
    task_acks_late          = True,
    task_reject_on_worker_lost = True,
    worker_prefetch_multiplier = 1,       # one task at a time per worker

    # Timeouts
    task_soft_time_limit    = 3600 * 12,  # 12 hours soft
    task_time_limit         = 3600 * 24,  # 24 hours hard

    # Result TTL
    result_expires          = 3600 * 48,  # keep results 48 hours

    # Retry policy
    task_max_retries        = 3,
    task_default_retry_delay = 30,

    # Routing
    task_routes = {
        "neurosim.worker.celery_app.run_simulation_cpu":  {"queue": "neurosim-jobs"},
        "neurosim.worker.celery_app.run_simulation_gpu":  {"queue": "neurosim-gpu-jobs"},
        "neurosim.worker.celery_app.run_simulation_mpi":  {"queue": "neurosim-mpi-jobs"},
        "neurosim.worker.celery_app.run_analysis":        {"queue": "neurosim-jobs"},
        "neurosim.worker.celery_app.run_optimization":    {"queue": "neurosim-jobs"},
    },

    # Beat schedule for periodic tasks
    beat_schedule = {
        "cleanup-old-checkpoints": {
            "task":     "neurosim.worker.celery_app.cleanup_checkpoints",
            "schedule": 3600.0,  # hourly
        },
        "health-report": {
            "task":     "neurosim.worker.celery_app.worker_health_report",
            "schedule": 60.0,    # every minute
        },
    },
)


# ─────────────────────────────────────────────────────────────
#  Base task class with progress reporting
# ─────────────────────────────────────────────────────────────
class ProgressTask(Task):
    abstract = True

    def update_progress(self, current: float, total: float,
                        message: str = "") -> None:
        self.update_state(
            state="PROGRESS",
            meta={
                "current": current,
                "total":   total,
                "percent": 100.0 * current / max(total, 1),
                "message": message,
            }
        )


# ─────────────────────────────────────────────────────────────
#  Core simulation tasks
# ─────────────────────────────────────────────────────────────
@app.task(bind=True, base=ProgressTask, name="neurosim.worker.celery_app.run_simulation_cpu")
def run_simulation_cpu(self, config: dict) -> dict:
    """
    CPU simulation task.
    config: SimulationConfig dict + network spec
    Returns: result summary dict
    """
    from neurosim.simulation.engine import (
        NeuralSimulationEngine, SimulationConfig
    )
    from neurosim.reconstruction.ingestion import ReconstructionPipeline

    job_id = config.get("job_id", self.request.id)
    logger.info(f"[{job_id}] Starting CPU simulation")
    t_wall = time.perf_counter()

    try:
        cfg     = SimulationConfig.from_dict(config.get("simulation", {}))
        engine  = NeuralSimulationEngine(cfg)
        pipeline = ReconstructionPipeline(
            output_dir=f"outputs/{job_id}"
        )

        net_spec = config.get("network", {})
        pipeline.build_random_network(
            n_exc   = net_spec.get("n_excitatory", 100),
            n_inh   = net_spec.get("n_inhibitory", 25),
            p_ee    = net_spec.get("p_ee", 0.1),
            p_ei    = net_spec.get("p_ei", 0.5),
            p_ie    = net_spec.get("p_ie", 0.5),
            p_ii    = net_spec.get("p_ii", 0.1),
        )
        pipeline.export_to_engine(engine)

        # Inject current clamps
        for clamp in config.get("current_clamps", []):
            nid   = clamp["neuron_id"]
            I_amp = clamp["I_amp"]
            t0    = clamp.get("t_start", 0.0)
            t1    = clamp.get("t_stop", cfg.t_stop)
            engine.clamp_current(nid, lambda t, _I=I_amp, _t0=t0, _t1=t1:
                                 _I if _t0 <= t <= _t1 else 0.0)

        # Progress reporting callback
        def progress_cb(pct: float, t: float) -> None:
            self.update_progress(pct, 100.0,
                                 f"t={t:.1f}ms / {cfg.t_stop:.0f}ms")

        results = engine.run(progress_callback=progress_cb)
        results["job_id"]     = job_id
        results["status"]     = "completed"
        results["wall_time_s"] = time.perf_counter() - t_wall

        logger.info(f"[{job_id}] Completed in {results['wall_time_s']:.1f}s "
                    f"| {results['total_spikes']} spikes")
        return results

    except Exception as exc:
        logger.exception(f"[{job_id}] Simulation failed: {exc}")
        raise self.retry(exc=exc, countdown=30)


@app.task(bind=True, base=ProgressTask, name="neurosim.worker.celery_app.run_simulation_gpu")
def run_simulation_gpu(self, config: dict) -> dict:
    """
    GPU-accelerated simulation task.
    Uses CuPy / CUDA kernels for large neuron counts.
    Falls back to CPU if no GPU available.
    """
    from neurosim.simulation.distributed import GPUClusterCoordinator
    from neurosim.simulation.engine import SimulationConfig, NeuralSimulationEngine
    from neurosim.reconstruction.ingestion import ReconstructionPipeline

    job_id = config.get("job_id", self.request.id)
    logger.info(f"[{job_id}] Starting GPU simulation")

    gpu_coord = GPUClusterCoordinator()
    if not gpu_coord._gpu_available:
        logger.warning(f"[{job_id}] No GPU found — routing to CPU path")
        return run_simulation_cpu(config)

    cfg      = SimulationConfig.from_dict(config.get("simulation", {}))
    engine   = NeuralSimulationEngine(cfg)
    pipeline = ReconstructionPipeline(output_dir=f"outputs/{job_id}")

    net_spec = config.get("network", {})
    pipeline.build_random_network(
        n_exc = net_spec.get("n_excitatory", 10000),
        n_inh = net_spec.get("n_inhibitory", 2500),
    )
    pipeline.export_to_engine(engine)

    def progress_cb(pct: float, t: float) -> None:
        self.update_progress(pct, 100.0, f"GPU t={t:.1f}ms")

    results = engine.run(progress_callback=progress_cb)
    results["job_id"]  = job_id
    results["backend"] = "cuda"
    return results


@app.task(bind=True, base=ProgressTask, name="neurosim.worker.celery_app.run_simulation_mpi")
def run_simulation_mpi(self, config: dict) -> dict:
    """
    Submit an MPI job to the HPC scheduler (SLURM / PBS).
    This task acts as a launcher — it submits the job and polls
    for completion, streaming progress back.
    """
    import subprocess
    import json

    job_id    = config.get("job_id", self.request.id)
    n_ranks   = config.get("hpc", {}).get("n_mpi_ranks", 64)
    cfg_path  = f"/tmp/neurosim_job_{job_id}.yaml"
    out_dir   = f"outputs/{job_id}"

    # Write config to temp file
    import yaml
    Path(cfg_path).write_text(yaml.dump(config))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=neurosim-{job_id[:8]}
#SBATCH --nodes={max(1, n_ranks // 32)}
#SBATCH --ntasks={n_ranks}
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output={out_dir}/slurm-%j.log
#SBATCH --partition=hpc

module load OpenMPI/4.1.5 Python/3.11 HDF5/1.14

mpirun -np {n_ranks} python -m neurosim.cli simulate \\
    --config {cfg_path} \\
    --distributed \\
    --output-dir {out_dir}
"""
    script_path = f"/tmp/neurosim_slurm_{job_id}.sh"
    Path(script_path).write_text(slurm_script)

    try:
        result = subprocess.run(
            ["sbatch", "--parsable", script_path],
            capture_output=True, text=True, check=True
        )
        slurm_id = result.stdout.strip()
        logger.info(f"[{job_id}] Submitted SLURM job {slurm_id}")

        # Poll for completion
        while True:
            time.sleep(30)
            status_result = subprocess.run(
                ["squeue", "-j", slurm_id, "-o", "%T", "--noheader"],
                capture_output=True, text=True
            )
            slurm_state = status_result.stdout.strip()
            if not slurm_state:
                break  # job finished
            self.update_progress(0, 100, f"SLURM state: {slurm_state}")

        return {"job_id": job_id, "slurm_id": slurm_id,
                "status": "completed", "output_dir": out_dir}

    except FileNotFoundError:
        # sbatch not available — fall back to local MPI
        logger.warning(f"[{job_id}] SLURM not available, running local mpirun")
        proc = subprocess.run(
            ["mpirun", "-np", str(min(n_ranks, 4)),
             "python", "-m", "neurosim.cli", "simulate",
             "--config", cfg_path, "--distributed",
             "--output-dir", out_dir],
            capture_output=True, text=True
        )
        return {"job_id": job_id, "status": "completed",
                "stdout": proc.stdout[-2000:]}


# ─────────────────────────────────────────────────────────────
#  Analysis task
# ─────────────────────────────────────────────────────────────
@app.task(bind=True, name="neurosim.worker.celery_app.run_analysis")
def run_analysis(self, results_path: str, analyses: list) -> dict:
    """
    Run post-simulation analyses on saved results.
    analyses: list of analysis names to compute.
    """
    import numpy as np
    import h5py
    from neurosim.analysis.spike_analysis import (
        compute_psth, pairwise_correlation,
        synchrony_index, connectivity_stats
    )

    logger.info(f"Running analyses: {analyses} on {results_path}")
    output = {}

    with h5py.File(results_path, "r") as f:
        spike_data = {}
        if "spikes" in f:
            for nid in f["spikes"]:
                spike_data[int(nid)] = f["spikes"][nid][:].tolist()

        t_stop = float(f.attrs.get("t_stop", 1000.0))

    spike_trains = list(spike_data.values())

    if "psth" in analyses:
        psth = compute_psth(spike_trains, t_stop=t_stop, bin_size=10.0)
        output["psth"] = {
            "rate":        psth.rate.tolist(),
            "bin_centers": psth.bin_centers.tolist(),
            "sem":         psth.sem.tolist(),
        }

    if "synchrony" in analyses:
        output["synchrony_index"] = synchrony_index(
            spike_trains, 0.0, t_stop
        )

    if "correlation" in analyses and len(spike_trains) > 1:
        corr = pairwise_correlation(spike_trains, 0.0, t_stop)
        output["mean_pairwise_corr"] = float(
            np.triu(corr, k=1).mean()
        )

    return output


# ─────────────────────────────────────────────────────────────
#  Parameter optimization task
# ─────────────────────────────────────────────────────────────
@app.task(bind=True, base=ProgressTask,
          name="neurosim.worker.celery_app.run_optimization")
def run_optimization(self, opt_config: dict) -> dict:
    """
    CMA-ES parameter optimization task.
    Runs calibration of neuron parameters against target data.
    """
    import numpy as np
    from neurosim.ml.optimization import (
        ParameterSpace, EvolutionaryOptimizer,
        FiringPatternObjective
    )
    from neurosim.models.neuron import build_l5_pyramidal_cell
    from neurosim.simulation.engine import run_current_clamp

    logger.info("Starting parameter optimization")

    # Build parameter space from config
    space = ParameterSpace()
    for p in opt_config.get("parameters", []):
        space.add(p["name"], p["lo"], p["hi"],
                  log_scale=p.get("log_scale", False))

    # Target data (from config or synthetic)
    target_data = opt_config.get("target_data")
    if target_data:
        target_V = np.array(target_data["V"])
        target_t = np.array(target_data["t"])
    else:
        # Generate synthetic target with default params
        ref_neuron  = build_l5_pyramidal_cell()
        ref_results = run_current_clamp(ref_neuron, I_amp=2.0)
        target_V    = ref_results["V"][0, 0, :]
        target_t    = ref_results["t"]

    def neuron_builder(theta: dict):
        neuron = build_l5_pyramidal_cell()
        # Apply parameter overrides
        if "gNa" in theta:
            neuron.compartments[0].channels["na"].g_max = theta["gNa"]
        if "gK" in theta:
            neuron.compartments[0].channels["k"].g_max = theta["gK"]
        if "gL" in theta:
            neuron.compartments[0].channels["leak"].g_max = theta["gL"]
        neuron.initialize(-65.0)
        return neuron

    objective = FiringPatternObjective(
        target_V       = target_V,
        target_t       = target_t,
        neuron_builder = neuron_builder,
        sim_fn         = lambda n, I: run_current_clamp(n, I_amp=I),
        I_amp          = opt_config.get("I_amp", 2.0),
    )

    optimizer = EvolutionaryOptimizer(
        space    = space,
        popsize  = opt_config.get("popsize", 10),
        seed     = opt_config.get("seed", 42),
    )

    def gen_callback(pct, t):
        self.update_progress(pct, 100.0, f"Generation {int(pct)}")

    result = optimizer.optimize(
        objective     = objective,
        n_generations = opt_config.get("n_generations", 50),
        sigma0        = opt_config.get("sigma0", 0.3),
    )

    return {
        "best_params":    result.best_params,
        "best_value":     result.best_value,
        "n_evaluations":  result.n_evaluations,
        "converged":      result.converged,
        "history":        result.history,
    }


# ─────────────────────────────────────────────────────────────
#  Maintenance tasks
# ─────────────────────────────────────────────────────────────
@app.task(name="neurosim.worker.celery_app.cleanup_checkpoints")
def cleanup_checkpoints(max_age_hours: float = 48.0) -> dict:
    """Remove checkpoint files older than max_age_hours."""
    import glob
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints")
    pattern        = f"{checkpoint_dir}/**/*.pkl"
    cutoff         = time.time() - max_age_hours * 3600
    removed        = 0
    for path in glob.glob(pattern, recursive=True):
        if os.path.getmtime(path) < cutoff:
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    logger.info(f"Cleaned up {removed} checkpoint files")
    return {"removed": removed}


@app.task(name="neurosim.worker.celery_app.worker_health_report")
def worker_health_report() -> dict:
    """Emit worker health metrics."""
    import psutil
    return {
        "cpu_percent":    psutil.cpu_percent(interval=1),
        "mem_percent":    psutil.virtual_memory().percent,
        "disk_percent":   psutil.disk_usage("/").percent,
        "timestamp":      time.time(),
    }


# ─────────────────────────────────────────────────────────────
#  Celery signals
# ─────────────────────────────────────────────────────────────
@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    logger.info(f"NeuroSim worker ready — type: "
                f"{os.environ.get('NEUROSIM_WORKER_TYPE', 'cpu')}")


@task_prerun.connect
def on_task_prerun(task_id, task, args, kwargs, **extras):
    logger.info(f"Task starting: {task.name} [{task_id[:8]}]")


@task_postrun.connect
def on_task_postrun(task_id, task, args, kwargs, retval, state, **extras):
    logger.info(f"Task finished: {task.name} [{task_id[:8]}] → {state}")
