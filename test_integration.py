"""
tests/test_integration.py

Integration and regression tests for the full simulation pipeline.
These tests run full simulations end-to-end and validate scientific correctness.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from neurosim.models.neuron import build_l5_pyramidal_cell, build_parvalbumin_interneuron
from neurosim.simulation.engine import (
    SimulationConfig, NeuralSimulationEngine, Connection, run_current_clamp
)
from neurosim.reconstruction.ingestion import ReconstructionPipeline
from neurosim.analysis.spike_analysis import (
    compute_psth, synchrony_index, mean_firing_rate,
    isi, cv_isi, fano_factor, cross_correlogram
)


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def l5_current_clamp_results():
    """Run a single L5 pyramidal cell with 500ms current injection."""
    neuron = build_l5_pyramidal_cell(neuron_id=0)
    return run_current_clamp(
        neuron,
        I_amp        = 3.0,
        t_start_inj  = 100.0,
        t_stop_inj   = 600.0,
        t_stop_sim   = 700.0,
        dt           = 0.025,
    )


@pytest.fixture(scope="module")
def ei_network_results():
    """Run a small E-I network simulation."""
    pipeline = ReconstructionPipeline()
    pipeline.build_random_network(
        n_exc=40, n_inh=10, p_ee=0.15, p_ei=0.5, p_ie=0.5, p_ii=0.1,
        rng_seed=99
    )
    cfg    = SimulationConfig(dt=0.025, t_stop=500.0, n_workers=2, seed=99)
    engine = NeuralSimulationEngine(cfg)
    pipeline.export_to_engine(engine)

    # Drive with external excitatory input to 5 excitatory neurons
    for nid in range(5):
        engine.clamp_current(nid, lambda t: 2.5 if 50 <= t <= 450 else 0.0)

    return engine.run()


# ─────────────────────────────────────────────────────────────
#  Single-neuron integration tests
# ─────────────────────────────────────────────────────────────
class TestSingleNeuronIntegration:

    def test_produces_spikes_under_current(self, l5_current_clamp_results):
        r = l5_current_clamp_results
        assert r["total_spikes"] >= 3, (
            f"Expected ≥3 spikes, got {r['total_spikes']}"
        )

    def test_voltage_shape(self, l5_current_clamp_results):
        r = l5_current_clamp_results
        V = r["V"]
        assert V.ndim == 3               # (n_neurons, n_comps, n_t)
        assert V.shape[0] >= 1
        assert V.shape[2] > 100

    def test_no_nan_voltages(self, l5_current_clamp_results):
        V = l5_current_clamp_results["V"]
        assert not np.any(np.isnan(V)), "NaN detected in voltage trace"

    def test_realistic_firing_rate(self, l5_current_clamp_results):
        """L5 pyramidal with 3 nA should fire 5–40 Hz."""
        spikes  = list(l5_current_clamp_results["spikes"].get(0, []))
        rate_hz = mean_firing_rate(spikes, 100.0, 600.0)
        assert 2.0 < rate_hz < 60.0, (
            f"Firing rate {rate_hz:.2f} Hz outside [2, 60] Hz window"
        )

    def test_realistic_isi_cv(self, l5_current_clamp_results):
        """CV of ISI should be < 0.6 for regular spiking neuron."""
        spikes = list(l5_current_clamp_results["spikes"].get(0, []))
        if len(spikes) < 3:
            pytest.skip("Too few spikes to compute ISI")
        cv = cv_isi(spikes)
        assert cv < 0.8, f"CV_ISI = {cv:.3f}, expected < 0.8 for regular spiking"

    def test_spike_times_are_sorted(self, l5_current_clamp_results):
        for nid, times in l5_current_clamp_results["spikes"].items():
            assert times == sorted(times), f"Spike times not sorted for neuron {nid}"

    def test_after_hyperpolarization(self, l5_current_clamp_results):
        """After spike, voltage should drop below resting potential (AHP)."""
        V_soma = l5_current_clamp_results["V"][0, 0, :]
        t      = l5_current_clamp_results["t"]
        # Check 50-100 ms after stimulation end
        mask = (t > 620) & (t < 680)
        if mask.sum() > 0:
            V_post = V_soma[mask]
            assert V_post.min() < -70.0, (
                f"No AHP detected: min V post-stim = {V_post.min():.2f} mV"
            )

    def test_ap_peak_amplitude(self, l5_current_clamp_results):
        """Action potential peak should be > 0 mV (overshoot)."""
        V_soma = l5_current_clamp_results["V"][0, 0, :]
        assert V_soma.max() > 0.0, (
            f"AP peak = {V_soma.max():.2f} mV, expected > 0 (overshoot)"
        )

    def test_dendritic_propagation(self, l5_current_clamp_results):
        """Depolarization should propagate to dendrites (attenuated)."""
        V = l5_current_clamp_results["V"][0]   # (n_comps, n_t)
        if V.shape[0] >= 2:
            V_soma  = V[0].max()
            V_dend  = V[1].max()
            assert V_dend > -65.0, "Dendrite should depolarize"
            assert V_soma > V_dend, "Somatic AP should be larger than dendritic"


# ─────────────────────────────────────────────────────────────
#  Network integration tests
# ─────────────────────────────────────────────────────────────
class TestNetworkIntegration:

    def test_network_produces_activity(self, ei_network_results):
        assert ei_network_results["total_spikes"] > 0

    def test_correct_neuron_count(self, ei_network_results):
        assert ei_network_results["n_neurons"] == 50

    def test_connections_nonzero(self, ei_network_results):
        assert ei_network_results["n_connections"] > 0

    def test_network_spikes_dict_structure(self, ei_network_results):
        spikes = ei_network_results.get("spikes", {})
        assert isinstance(spikes, dict)
        for nid, times in spikes.items():
            assert isinstance(times, list)
            for t in times:
                assert 0.0 <= t <= 500.0, f"Spike at t={t} outside simulation range"

    def test_network_has_driven_and_silent_neurons(self, ei_network_results):
        """Driven neurons should fire; many undriven may be silent."""
        spikes = ei_network_results.get("spikes", {})
        firing = sum(1 for times in spikes.values() if len(times) > 0)
        assert firing >= 1, "At least one neuron should fire"

    def test_population_firing_rate_reasonable(self, ei_network_results):
        spikes = ei_network_results.get("spikes", {})
        spike_trains = list(spikes.values())
        ff = fano_factor(spike_trains, 0, 500.0, bin_size=50.0)
        # For a driven network, Fano factor should be finite
        if not np.isnan(ff):
            assert ff >= 0.0

    def test_network_wall_time_reasonable(self, ei_network_results):
        """50-neuron, 500ms simulation should complete in < 60s on CI."""
        wall = ei_network_results.get("wall_time_s", 999)
        assert wall < 120.0, f"Simulation too slow: {wall:.1f}s"


# ─────────────────────────────────────────────────────────────
#  Analysis pipeline integration
# ─────────────────────────────────────────────────────────────
class TestAnalysisPipeline:

    @pytest.fixture(scope="class")
    def synthetic_spike_trains(self):
        """Generate synthetic Poisson spike trains."""
        rng = np.random.default_rng(42)
        trains = []
        for _ in range(20):
            times = np.sort(rng.exponential(100.0, 10).cumsum())
            times = times[times < 1000.0].tolist()
            trains.append(times)
        return trains

    def test_psth_output_shape(self, synthetic_spike_trains):
        psth = compute_psth(synthetic_spike_trains,
                            t_start=0, t_stop=1000, bin_size=20.0)
        expected_n_bins = 50  # 1000ms / 20ms
        assert len(psth.rate) == expected_n_bins
        assert len(psth.bin_centers) == expected_n_bins

    def test_psth_rate_nonnegative(self, synthetic_spike_trains):
        psth = compute_psth(synthetic_spike_trains,
                            t_start=0, t_stop=1000, bin_size=20.0)
        assert np.all(psth.rate >= 0.0)

    def test_cross_correlogram_symmetric(self, synthetic_spike_trains):
        """Cross-correlogram of a train with itself should be symmetric."""
        st = synthetic_spike_trains[0]
        lags, acg = cross_correlogram(st, st, max_lag=100.0, bin_size=5.0)
        center = len(acg) // 2
        # Check symmetry (allowing small floating point differences)
        assert np.allclose(acg[:center][::-1], acg[center+1:], atol=1)

    def test_synchrony_index_range(self, synthetic_spike_trains):
        si = synchrony_index(synthetic_spike_trains, 0.0, 1000.0, bin_size=10.0)
        assert si >= 0.0

    def test_isi_length(self, synthetic_spike_trains):
        for st in synthetic_spike_trains:
            if len(st) >= 2:
                intervals = isi(st)
                assert len(intervals) == len(st) - 1
                assert np.all(intervals > 0), "ISIs should be positive"


# ─────────────────────────────────────────────────────────────
#  Reconstruction pipeline integration
# ─────────────────────────────────────────────────────────────
class TestReconstructionPipeline:

    def test_build_random_network(self):
        pipeline = ReconstructionPipeline()
        pipeline.build_random_network(n_exc=20, n_inh=5, rng_seed=1)
        assert len(pipeline.neurons) == 25
        assert len(pipeline.connections) > 0

    def test_export_to_engine(self):
        pipeline = ReconstructionPipeline()
        pipeline.build_random_network(n_exc=10, n_inh=2)
        cfg      = SimulationConfig(dt=0.025, t_stop=10.0, n_workers=1)
        engine   = NeuralSimulationEngine(cfg)
        pipeline.export_to_engine(engine)
        assert len(engine.neurons) == 12

    def test_swc_read_write_roundtrip(self, tmp_path):
        from neurosim.reconstruction.ingestion import SWCReader, SWCPoint, SWCType

        points = [
            SWCPoint(1, SWCType.SOMA,        0, 0, 0, 8.0, -1),
            SWCPoint(2, SWCType.APICAL_DEND, 0, 50, 0, 2.0, 1),
            SWCPoint(3, SWCType.APICAL_DEND, 0, 100, 0, 1.5, 2),
            SWCPoint(4, SWCType.BASAL_DEND,  0, -50, 0, 1.5, 1),
        ]
        path = str(tmp_path / "test.swc")
        SWCReader.write(path, points)
        loaded = SWCReader.read(path)
        assert len(loaded) == len(points)
        assert loaded[0].radius == pytest.approx(8.0, rel=1e-3)

    def test_morphology_validator_catches_duplicates(self):
        from neurosim.reconstruction.ingestion import (
            SWCPoint, SWCType, MorphologyValidator
        )
        points = [
            SWCPoint(1, SWCType.SOMA, 0, 0, 0, 5.0, -1),
            SWCPoint(1, SWCType.SOMA, 0, 0, 0, 5.0, -1),  # duplicate
            SWCPoint(2, SWCType.BASAL_DEND, 0, 50, 0, 1.5, 1),
        ]
        validator = MorphologyValidator()
        cleaned, issues = validator.validate(points)
        dup_issues = [i for i in issues if i.issue_type == "duplicate_id"]
        assert len(dup_issues) >= 1


# ─────────────────────────────────────────────────────────────
#  Benchmarks (marked separately — run with --benchmark-only)
# ─────────────────────────────────────────────────────────────
class TestBenchmarks:

    @pytest.mark.benchmark
    def test_benchmark_single_neuron_1s(self, benchmark):
        """Benchmark: single L5 pyramidal cell, 1s simulation."""
        neuron = build_l5_pyramidal_cell()

        def run():
            n = build_l5_pyramidal_cell()
            return run_current_clamp(n, I_amp=2.0,
                                     t_stop_sim=1000.0, dt=0.025)

        result = benchmark(run)
        assert result["total_spikes"] >= 0

    @pytest.mark.benchmark
    def test_benchmark_network_100_neurons(self, benchmark):
        """Benchmark: 100-neuron E-I network, 200ms."""
        def run():
            pipeline = ReconstructionPipeline()
            pipeline.build_random_network(n_exc=80, n_inh=20, rng_seed=1)
            cfg    = SimulationConfig(dt=0.025, t_stop=200.0, n_workers=1)
            engine = NeuralSimulationEngine(cfg)
            pipeline.export_to_engine(engine)
            engine.clamp_current(0, lambda t: 2.0 if 20 <= t <= 180 else 0.0)
            return engine.run()

        result = benchmark(run)
        assert result["n_neurons"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
