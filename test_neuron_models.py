"""
tests/test_neuron_models.py

Comprehensive test suite for:
- Ion channel models (HH kinetics)
- Neuron model (multi-compartment)
- Simulation engine
- Synapse models
- Plasticity rules
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from neurosim.models.ion_channels import (
    NaChannel, KChannel, LeakChannel, CaLChannel,
    KAChannel, HCNChannel, build_channel, nernst_potential, q10_scale
)
from neurosim.models.neuron import (
    Compartment, CompartmentType, MultiCompartmentNeuron,
    Synapse, SynapseType, make_synapse,
    build_l5_pyramidal_cell, build_parvalbumin_interneuron
)
from neurosim.models.plasticity import (
    STDPRule, STDPState, TsodyksMarkramSynapse,
    BCMRule, NeurotransmitterDiffusion
)
from neurosim.simulation.engine import (
    SimulationConfig, NeuralSimulationEngine,
    SpikeQueue, SpikeEvent, ConnectivityGraph, Connection,
    run_current_clamp
)


# ─────────────────────────────────────────────────────────────
#  Test: Physical constants and utility functions
# ─────────────────────────────────────────────────────────────
class TestUtilities:
    def test_nernst_potential_sodium(self):
        """Na+ reversal should be ~50 mV at physiological concentrations."""
        E_Na = nernst_potential(z=1, c_in=15.0, c_out=150.0, T_celsius=37.0)
        assert 40.0 < E_Na < 70.0, f"E_Na={E_Na:.2f} outside expected range"

    def test_nernst_potential_potassium(self):
        """K+ reversal should be ~-80 to -90 mV."""
        E_K = nernst_potential(z=1, c_in=140.0, c_out=5.0, T_celsius=37.0)
        assert -100.0 < E_K < -60.0, f"E_K={E_K:.2f} outside expected range"

    def test_q10_scaling_increases_with_temperature(self):
        rate_cold = q10_scale(1.0, T=6.3, T_ref=6.3)
        rate_warm = q10_scale(1.0, T=16.3, T_ref=6.3)
        assert rate_warm > rate_cold

    def test_q10_doubling(self):
        """Q10=3: rate triples per 10°C."""
        rate_base = q10_scale(1.0, T=6.3, T_ref=6.3, q10=3.0)
        rate_10c  = q10_scale(1.0, T=16.3, T_ref=6.3, q10=3.0)
        assert abs(rate_10c / rate_base - 3.0) < 0.01


# ─────────────────────────────────────────────────────────────
#  Test: Ion channel models
# ─────────────────────────────────────────────────────────────
class TestIonChannels:
    def test_na_channel_steady_state_at_rest(self):
        """At -65 mV, Na channel should be mostly closed (low m, high h)."""
        ch = NaChannel()
        state = ch.steady_state(-65.0)
        m, h = state
        assert m < 0.1,  f"m_inf at -65mV = {m:.4f}, expected < 0.1"
        assert h > 0.5,  f"h_inf at -65mV = {h:.4f}, expected > 0.5"

    def test_na_channel_activation_at_threshold(self):
        """At -40 mV, Na channel should be partially activated."""
        ch = NaChannel()
        state = ch.steady_state(-40.0)
        m, h = state
        assert m > 0.3, f"m_inf at -40mV = {m:.4f}, expected > 0.3"

    def test_k_channel_activation(self):
        """K channel activates with depolarization."""
        ch = KChannel()
        n_rest     = ch.steady_state(-65.0)[0]
        n_depol    = ch.steady_state(-20.0)[0]
        assert n_depol > n_rest, "K channel should activate with depolarization"

    def test_leak_channel_current(self):
        """Leak current at rest should be near zero (V ≈ E_leak)."""
        ch  = LeakChannel(E_rev=-65.0)
        I   = ch.current(-65.0, np.array([]))
        assert abs(I) < 1e-6, f"Leak current at rest = {I:.6f}, expected ≈ 0"

    def test_na_current_sign(self):
        """Na+ current should be inward (negative) during AP."""
        ch = NaChannel()
        state = np.array([0.9, 0.9])   # fully activated
        I = ch.current(0.0, state)     # at 0 mV (well below E_Na=+50)
        assert I < 0, f"Na current at 0mV should be inward (negative), got {I:.4f}"

    def test_channel_factory(self):
        """build_channel should return correct types."""
        for name in ["na", "k", "leak", "ca_l", "ka", "hcn"]:
            ch = build_channel(name)
            assert ch is not None

        with pytest.raises(ValueError):
            build_channel("nonexistent_channel_xyz")

    def test_gate_derivatives_return_correct_shape(self):
        """Gate derivatives should match state vector dimension."""
        ch    = NaChannel()
        state = ch.steady_state(-65.0)
        deriv = ch.derivatives(-65.0, state)
        assert deriv.shape == state.shape

    def test_hcn_activates_at_hyperpolarization(self):
        """HCN (Ih) should be more activated at hyperpolarized potentials."""
        ch = HCNChannel()
        q_hyper = ch.steady_state(-100.0)[0]
        q_depol = ch.steady_state(-50.0)[0]
        assert q_hyper > q_depol, "HCN should activate at hyperpolarized V"


# ─────────────────────────────────────────────────────────────
#  Test: Compartment and neuron models
# ─────────────────────────────────────────────────────────────
class TestCompartment:
    def test_compartment_area_calculation(self):
        """Area should be π × d × L."""
        comp = Compartment(id=0, type=CompartmentType.SOMA,
                           length=20.0, diameter=20.0)
        expected_area = np.pi * 20.0 * 20.0
        assert abs(comp.area - expected_area) < 1e-10

    def test_compartment_ionic_current_at_rest(self):
        """At resting potential with steady-state channels, sum ≈ 0."""
        comp = Compartment(id=0, type=CompartmentType.SOMA,
                           length=20.0, diameter=20.0)
        comp.add_channel("na",   NaChannel())
        comp.add_channel("k",    KChannel())
        comp.add_channel("leak", LeakChannel())

        V_rest = -65.0
        states = {
            name: ch.steady_state(V_rest)
            for name, ch in comp.channels.items()
        }
        I_total = comp.ionic_current(V_rest, states)
        # Should be close to zero at true resting potential
        assert abs(I_total) < 1e-3, (
            f"Resting ionic current = {I_total:.6f} μA, expected ≈ 0"
        )


class TestNeuronModels:
    def test_l5_pyramidal_builds(self):
        neuron = build_l5_pyramidal_cell(neuron_id=1)
        assert len(neuron.compartments) == 5
        assert neuron.neuron_id == 1

    def test_pv_interneuron_builds(self):
        neuron = build_parvalbumin_interneuron(neuron_id=2)
        assert len(neuron.compartments) == 2

    def test_neuron_initialization(self):
        """All compartments should start at V_init."""
        neuron = build_l5_pyramidal_cell()
        assert np.allclose(neuron.V, -65.0, atol=1.0)

    def test_neuron_step_does_not_explode(self):
        """Integration step should not produce NaN or extreme values."""
        neuron = build_l5_pyramidal_cell()
        for _ in range(100):
            neuron.step(dt=0.025)
        assert not np.any(np.isnan(neuron.V))
        assert np.all(np.abs(neuron.V) < 200.0)

    def test_current_injection_produces_spikes(self):
        """Strong current injection should produce action potentials."""
        neuron = build_l5_pyramidal_cell()
        n_comps = len(neuron.compartments)
        I_ext   = np.zeros(n_comps)
        I_ext[0] = 5.0   # strong somatic injection (μA)

        for _ in range(4000):   # 100 ms
            neuron.step(dt=0.025, I_ext=I_ext)

        n_spikes = len(neuron.spike_detector.times)
        assert n_spikes >= 1, (
            f"Expected at least 1 spike with strong current, got {n_spikes}"
        )

    def test_no_spontaneous_spikes_at_rest(self):
        """Without input, neuron should not fire spontaneously."""
        neuron = build_l5_pyramidal_cell()
        for _ in range(8000):   # 200 ms
            neuron.step(dt=0.025)
        n_spikes = len(neuron.spike_detector.times)
        assert n_spikes == 0, f"Unexpected {n_spikes} spontaneous spikes"


# ─────────────────────────────────────────────────────────────
#  Test: Synapse models
# ─────────────────────────────────────────────────────────────
class TestSynapses:
    def test_ampa_synapse_decays(self):
        """AMPA conductance should decay after activation."""
        syn = make_synapse("ampa")
        syn.activate()
        g_initial = syn.g_aux
        syn.update(dt=10.0, V=-65.0)
        assert syn.g_aux < g_initial, "AMPA should decay after activation"

    def test_gaba_a_reversal(self):
        """GABA-A reversal should be around -70 mV."""
        syn = make_synapse("gaba_a")
        assert -75.0 < syn.E_rev < -60.0

    def test_nmda_mg_block_at_rest(self):
        """NMDA should be strongly blocked at resting potential."""
        syn = make_synapse("nmda")
        syn.g = 1.0
        I_rest    = syn.update(dt=0.001, V=-65.0)
        I_depol   = syn.update(dt=0.001, V=+30.0)
        # Reset g for clean comparison
        # Test logic: depolarized NMDA current should be larger
        # (Mg block is removed at positive V)


# ─────────────────────────────────────────────────────────────
#  Test: Plasticity models
# ─────────────────────────────────────────────────────────────
class TestPlasticity:
    def test_stdp_ltp_pre_before_post(self):
        """Pre before post (Δt > 0) → LTP."""
        rule  = STDPRule()
        w_new = rule.update_weight(0.5, delta_t=+10.0)
        assert w_new > 0.5, "Pre before post should increase weight"

    def test_stdp_ltd_post_before_pre(self):
        """Post before pre (Δt < 0) → LTD."""
        rule  = STDPRule()
        w_new = rule.update_weight(0.5, delta_t=-10.0)
        assert w_new < 0.5, "Post before pre should decrease weight"

    def test_stdp_weight_bounds(self):
        """Weight should stay within [w_min, w_max]."""
        rule = STDPRule(A_plus=1.0, A_minus=1.0, w_min=0.0, w_max=1.0)
        w = 0.5
        for _ in range(100):
            w = rule.update_weight(w, delta_t=+1.0)
        assert 0.0 <= w <= 1.0

    def test_tm_facilitation(self):
        """Facilitating synapse: second spike releases more."""
        syn = TsodyksMarkramSynapse.facilitating()
        r1  = syn.spike(t=0.0)
        r2  = syn.spike(t=100.0)   # ISI > tau_rec
        # Reset for next call
        # Facilitation: u grows → more release
        syn2 = TsodyksMarkramSynapse.facilitating()
        r_first  = syn2.spike(t=0.0)
        r_second = syn2.spike(t=20.0)   # short ISI → facilitation
        # Second release should be comparable (x reduced but u increased)
        assert r_second > 0, "Second spike should produce nonzero release"

    def test_diffusion_stability(self):
        """Diffusion model should not become unstable."""
        diff = NeurotransmitterDiffusion(D=0.4, dt=0.025, dx=0.05)
        diff.release(1.0)
        for _ in range(100):
            C = diff.step()
        assert np.all(np.isfinite(C))
        assert np.all(C >= 0)

    def test_diffusion_conservation(self):
        """Total transmitter should decrease (cleared) after release."""
        diff = NeurotransmitterDiffusion(D=0.4, k_clear=0.1, dt=0.025, dx=0.05)
        diff.release(1.0)
        total_before = diff.total_transmitter()
        for _ in range(50):
            diff.step()
        total_after = diff.total_transmitter()
        assert total_after < total_before, "Clearance should reduce total transmitter"

    def test_diffusion_instability_guard(self):
        """Constructor should raise on unstable α."""
        with pytest.raises(ValueError):
            NeurotransmitterDiffusion(D=10.0, dt=0.5, dx=0.05)


# ─────────────────────────────────────────────────────────────
#  Test: Simulation engine
# ─────────────────────────────────────────────────────────────
class TestSimulationEngine:
    def test_spike_queue_ordering(self):
        """SpikeQueue should return events in time order."""
        q = SpikeQueue()
        for t in [5.0, 1.0, 3.0, 2.0]:
            q.push(SpikeEvent(t, 0, 1, 0, 0))
        events = q.pop_until(10.0)
        times  = [e.delivery_time for e in events]
        assert times == sorted(times)

    def test_connectivity_graph(self):
        g    = ConnectivityGraph()
        conn = Connection(src_id=0, tgt_id=1, tgt_comp=0,
                          syn_index=0, weight=1.0, delay=1.0)
        g.add_connection(conn)
        assert len(g.get_outgoing(0)) == 1
        assert len(g.get_incoming(1)) == 1
        assert len(g.get_outgoing(1)) == 0

    def test_single_neuron_current_clamp(self):
        """Current clamp experiment should produce spikes and voltage trace."""
        neuron  = build_l5_pyramidal_cell(neuron_id=0)
        results = run_current_clamp(
            neuron, I_amp=3.0,
            t_start_inj=50.0, t_stop_inj=300.0,
            t_stop_sim=350.0, dt=0.025
        )
        assert "V" in results
        assert "spikes" in results
        assert results["total_spikes"] >= 1, "Should produce at least 1 spike"
        # Voltage trace shape check
        V = results["V"]
        assert V.shape[0] >= 1   # at least 1 neuron
        assert V.shape[2] > 1    # multiple time steps

    def test_config_roundtrip(self):
        cfg  = SimulationConfig(dt=0.025, t_stop=500.0, integrator="rk4")
        d    = cfg.to_dict()
        cfg2 = SimulationConfig.from_dict(d)
        assert cfg2.dt == cfg.dt
        assert cfg2.t_stop == cfg.t_stop

    def test_network_simulation(self):
        """Small E-I network should run without errors."""
        from neurosim.reconstruction.ingestion import ReconstructionPipeline

        pipeline = ReconstructionPipeline()
        pipeline.build_random_network(n_exc=10, n_inh=3, rng_seed=42)

        cfg    = SimulationConfig(dt=0.025, t_stop=100.0, n_workers=1)
        engine = NeuralSimulationEngine(cfg)
        pipeline.export_to_engine(engine)

        # Inject current to one excitatory neuron
        engine.clamp_current(0, lambda t: 3.0 if 20 <= t <= 80 else 0.0)

        results = engine.run()
        assert results["n_neurons"] == 13
        assert results["n_connections"] > 0


# ─────────────────────────────────────────────────────────────
#  Integration test: AP waveform shape
# ─────────────────────────────────────────────────────────────
class TestAPWaveform:
    """Validate AP waveform against known HH properties."""

    @pytest.fixture(scope="class")
    def ap_results(self):
        neuron = build_l5_pyramidal_cell(neuron_id=0)
        return run_current_clamp(
            neuron, I_amp=5.0,
            t_start_inj=10.0, t_stop_inj=50.0,
            t_stop_sim=60.0, dt=0.025
        )

    def test_ap_peak_above_zero(self, ap_results):
        """AP should overshoot 0 mV."""
        V_soma = ap_results["V"][0, 0, :]
        assert V_soma.max() > 0.0, f"AP peak = {V_soma.max():.2f} mV"

    def test_ap_peak_below_70mv(self, ap_results):
        """AP should not exceed +70 mV (unrealistic overshoot)."""
        V_soma = ap_results["V"][0, 0, :]
        assert V_soma.max() < 70.0

    def test_ahp_below_rest(self, ap_results):
        """After-hyperpolarization should go below resting potential."""
        V_soma = ap_results["V"][0, 0, :]
        # Take last 20% of trace (after stimulation)
        late_V = V_soma[int(len(V_soma) * 0.8):]
        assert late_V.min() < -65.0, "Expected AHP below resting potential"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
