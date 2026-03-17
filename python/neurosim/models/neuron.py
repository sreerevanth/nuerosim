"""
neurosim/models/neuron.py

Multi-compartment neuron model with:
- Hodgkin–Huxley dynamics per compartment
- Cable equation for dendritic propagation
- Axon initial segment spike initiation
- Dendritic morphology (SWC-based)
- Calcium dynamics with buffering
- NMDA/AMPA/GABA synaptic inputs
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto
import warnings

from .ion_channels import (
    IonChannel, NaChannel, KChannel, LeakChannel,
    CaLChannel, KAChannel, KCaChannel, HCNChannel, build_channel
)

DEFAULT_T = 37.0  # °C  body temperature


# ─────────────────────────────────────────────────────────────
#  Compartment types
# ─────────────────────────────────────────────────────────────
class CompartmentType(Enum):
    SOMA         = auto()
    APICAL_DEND  = auto()
    BASAL_DEND   = auto()
    AXON_HILLOCK = auto()
    AXON_NODE    = auto()
    AXON_MYELIN  = auto()


# ─────────────────────────────────────────────────────────────
#  Compartment data structure
# ─────────────────────────────────────────────────────────────
@dataclass
class Compartment:
    """
    Single cylindrical compartment in a multi-compartment neuron.
    Implements cable equation: Cm * dV/dt = -I_ionic + I_axial + I_ext
    """
    id:         int
    type:       CompartmentType
    length:     float             # μm
    diameter:   float             # μm
    x: float = 0.0               # μm spatial position
    y: float = 0.0
    z: float = 0.0

    # Biophysical parameters
    Cm:   float = 1.0            # μF/cm²  specific membrane capacitance
    Ra:   float = 100.0          # Ω·cm    axial resistivity
    T:    float = 37.0           # °C

    # Channels on this compartment
    channels: Dict[str, IonChannel] = field(default_factory=dict)

    # Connectivity
    parent_id:   Optional[int] = None
    children_ids: List[int]    = field(default_factory=list)

    def __post_init__(self):
        # Derived geometric quantities
        self.area       = np.pi * self.diameter * self.length  # μm²  (lateral surface)
        self.area_cm2   = self.area * 1e-8                     # cm²
        self.cap        = self.Cm * self.area_cm2              # μF   total membrane cap
        self.r_axial    = (4 * self.Ra * self.length * 1e-4) / \
                          (np.pi * (self.diameter * 1e-4) ** 2)  # Ω

    def add_channel(self, name: str, channel: IonChannel) -> None:
        self.channels[name] = channel

    def ionic_current(self, V: float,
                      gate_states: Dict[str, np.ndarray]) -> float:
        """Sum of all ionic currents (μA/cm²) → scaled to μA."""
        I_total = 0.0
        for name, ch in self.channels.items():
            state = gate_states.get(name, ch.steady_state(V))
            I_total += ch.current(V, state)
        return I_total * self.area_cm2  # μA

    def gate_derivatives(self, V: float,
                         gate_states: Dict[str, np.ndarray]
                         ) -> Dict[str, np.ndarray]:
        return {
            name: ch.derivatives(V, gate_states.get(name, ch.steady_state(V)))
            for name, ch in self.channels.items()
        }


# ─────────────────────────────────────────────────────────────
#  Calcium dynamics (intracellular buffer + pump)
# ─────────────────────────────────────────────────────────────
@dataclass
class CalciumDynamics:
    """
    Single-pool calcium dynamics with:
    - Buffering (fast mobile buffer + slow fixed buffer)
    - Plasma membrane Ca2+-ATPase pump
    - Diffusion coupling between compartments
    """
    tau_ca:    float = 80.0      # ms  decay time constant
    ca_rest:   float = 1e-4      # mM  resting [Ca2+]
    ca:        float = field(init=False)
    depth:     float = 0.1       # μm  shell depth for volume calculation

    def __post_init__(self):
        self.ca = self.ca_rest

    def update(self, I_ca: float, area_cm2: float, dt: float) -> None:
        """
        Update intracellular [Ca2+] given calcium current.
        Uses Faraday: d[Ca]/dt = -I_Ca/(2*F*Vol) - (ca - ca_rest)/tau
        """
        from .ion_channels import FARADAY
        vol = area_cm2 * (self.depth * 1e-4)  # cm³
        influx = -I_ca / (2 * FARADAY * vol) * 1e3  # mM/ms
        decay  = -(self.ca - self.ca_rest) / self.tau_ca
        self.ca += (influx + decay) * dt
        self.ca  = max(self.ca, 0.0)


# ─────────────────────────────────────────────────────────────
#  Synapse models
# ─────────────────────────────────────────────────────────────
class SynapseType(Enum):
    AMPA   = auto()
    NMDA   = auto()
    GABA_A = auto()
    GABA_B = auto()

@dataclass
class Synapse:
    """
    Conductance-based synapse with double-exponential kinetics.
    I_syn = g(t) * (V - E_rev)
    g(t) = g_max * (exp(-t/tau2) - exp(-t/tau1))  (normalized)
    """
    syn_type:   SynapseType
    g_max:      float          # μS  peak conductance
    E_rev:      float          # mV  reversal potential
    tau_rise:   float          # ms  rise time constant
    tau_decay:  float          # ms  decay time constant
    delay:      float = 0.5    # ms  synaptic delay
    weight:     float = 1.0    # dimensionless weight

    # State
    g:     float = field(default=0.0, init=False)
    g_aux: float = field(default=0.0, init=False)  # auxiliary for double-exp

    # NMDA-specific Mg2+ block
    mg_conc:    float = 1.0    # mM

    def activate(self, weight_scale: float = 1.0) -> None:
        """Trigger synaptic event."""
        norm = 1.0 / (self.tau_decay - self.tau_rise) * \
               (self.tau_decay * self.tau_rise) * \
               np.log(self.tau_decay / self.tau_rise)
        self.g_aux += self.g_max * self.weight * weight_scale / norm

    def update(self, dt: float, V: float = -65.0) -> float:
        """Advance synapse state, return synaptic current (μA)."""
        self.g_aux -= self.g_aux / self.tau_rise  * dt
        self.g     += (-self.g / self.tau_decay + self.g_aux) * dt
        self.g      = max(self.g, 0.0)

        g_eff = self.g
        if self.syn_type == SynapseType.NMDA:
            # Mg2+ voltage-dependent block (Jahr & Stevens, 1990)
            mg_block = 1.0 / (1.0 + (self.mg_conc / 3.57) *
                               np.exp(-0.062 * V))
            g_eff *= mg_block

        return g_eff * (V - self.E_rev) * 1e-3  # μA


def make_synapse(syn_type: str, **kwargs) -> Synapse:
    """Factory for standard synapse types."""
    defaults = {
        "ampa":   dict(syn_type=SynapseType.AMPA,   g_max=1.0,  E_rev=0.0,
                       tau_rise=0.2, tau_decay=2.0),
        "nmda":   dict(syn_type=SynapseType.NMDA,   g_max=0.5,  E_rev=0.0,
                       tau_rise=2.0, tau_decay=65.0),
        "gaba_a": dict(syn_type=SynapseType.GABA_A, g_max=1.0,  E_rev=-70.0,
                       tau_rise=0.5, tau_decay=5.0),
        "gaba_b": dict(syn_type=SynapseType.GABA_B, g_max=0.2,  E_rev=-90.0,
                       tau_rise=15.0, tau_decay=150.0),
    }
    params = defaults.get(syn_type.lower(), {})
    params.update(kwargs)
    return Synapse(**params)


# ─────────────────────────────────────────────────────────────
#  Spike detection and recording
# ─────────────────────────────────────────────────────────────
class SpikeDetector:
    def __init__(self, threshold: float = -20.0):
        self.threshold = threshold
        self._above   = False
        self.times: List[float] = []

    def check(self, t: float, V: float) -> bool:
        fired = False
        if V > self.threshold and not self._above:
            self.times.append(t)
            self._above = True
            fired = True
        elif V < self.threshold:
            self._above = False
        return fired


# ─────────────────────────────────────────────────────────────
#  Multi-compartment neuron
# ─────────────────────────────────────────────────────────────
class MultiCompartmentNeuron:
    """
    Full multi-compartment neuron with:
    - Cable equation propagation
    - Per-compartment ion channels
    - Synaptic inputs
    - Calcium dynamics
    - Spike detection
    """

    def __init__(self, neuron_id: int, T: float = DEFAULT_T):
        self.neuron_id   = neuron_id
        self.T           = T
        self.compartments: List[Compartment]       = []
        self.synapses:    Dict[int, List[Synapse]] = {}  # comp_id → synapses
        self.ca_dynamics: Dict[int, CalciumDynamics] = {}
        self.spike_detector = SpikeDetector()

        # Simulation state
        self.V:     np.ndarray = np.array([])   # mV per compartment
        self.gates: List[Dict[str, np.ndarray]] = []
        self.t:     float = 0.0

    # ── Construction ──────────────────────────────────────────
    def add_compartment(self, comp: Compartment) -> None:
        self.compartments.append(comp)
        self.synapses[comp.id]    = []
        self.ca_dynamics[comp.id] = CalciumDynamics()

    def add_synapse(self, comp_id: int, syn: Synapse) -> None:
        self.synapses[comp_id].append(syn)

    def initialize(self, V_init: float = -65.0) -> None:
        """Set resting initial conditions."""
        n = len(self.compartments)
        self.V = np.full(n, V_init)
        self.gates = [
            {name: ch.steady_state(V_init)
             for name, ch in comp.channels.items()}
            for comp in self.compartments
        ]

    # ── Cable equation coupling ────────────────────────────────
    def _axial_currents(self) -> np.ndarray:
        """
        Compute axial (longitudinal) currents between compartments.
        I_axial(i→j) = (V_i - V_j) / R_couple
        """
        I_ax = np.zeros(len(self.compartments))
        comp_map = {c.id: i for i, c in enumerate(self.compartments)}
        for i, comp in enumerate(self.compartments):
            if comp.parent_id is not None:
                j  = comp_map[comp.parent_id]
                cp = self.compartments[j]
                r  = 0.5 * (comp.r_axial + cp.r_axial)  # Ω
                I  = (self.V[j] - self.V[i]) / r * 1e-3  # μA (V→mV, Ω→mS)
                I_ax[i] += I
                I_ax[j] -= I
        return I_ax

    # ── Main integration step ──────────────────────────────────
    def step(self, dt: float, I_ext: Optional[np.ndarray] = None) -> None:
        """
        Advance simulation by dt milliseconds using Forward Euler.
        For production use RK4 (see simulation_engine.py).
        """
        n = len(self.compartments)
        if I_ext is None:
            I_ext = np.zeros(n)

        I_ax = self._axial_currents()

        for i, comp in enumerate(self.compartments):
            V_i = self.V[i]

            # Ionic currents
            I_ion = comp.ionic_current(V_i, self.gates[i])

            # Synaptic currents
            I_syn = sum(s.update(dt, V_i) for s in self.synapses[comp.id])

            # Calcium update (if Ca channel present)
            if "ca_l" in comp.channels:
                I_ca = comp.channels["ca_l"].current(
                    V_i, self.gates[i].get("ca_l",
                         comp.channels["ca_l"].steady_state(V_i)))
                self.ca_dynamics[comp.id].update(
                    I_ca * comp.area_cm2, comp.area_cm2, dt)

            # Membrane voltage update: Cm * dV/dt = I_ax - I_ion - I_syn + I_ext
            dV = (I_ax[i] - I_ion - I_syn + I_ext[i]) / comp.cap
            self.V[i] += dV * dt

            # Update gate states
            dg = comp.gate_derivatives(V_i, self.gates[i])
            for name, deriv in dg.items():
                self.gates[i][name] += deriv * dt

        # Spike detection at soma (compartment 0)
        self.spike_detector.check(self.t, self.V[0])
        self.t += dt

    def get_state(self) -> Dict:
        return {
            "t":       self.t,
            "V":       self.V.copy(),
            "spikes":  list(self.spike_detector.times),
            "ca":      {cid: cd.ca for cid, cd in self.ca_dynamics.items()},
        }


# ─────────────────────────────────────────────────────────────
#  Default cell type builders
# ─────────────────────────────────────────────────────────────


def build_l5_pyramidal_cell(neuron_id: int = 0) -> MultiCompartmentNeuron:
    """
    Layer 5 thick-tufted pyramidal cell — simplified 5-compartment model.
    Compartments: soma, apical_prox, apical_dist, basal1, axon_initial
    """
    neuron = MultiCompartmentNeuron(neuron_id)

    soma = Compartment(id=0, type=CompartmentType.SOMA,
                       length=20.0, diameter=20.0)
    soma.add_channel("na",   NaChannel(g_max=120.0))
    soma.add_channel("k",    KChannel(g_max=36.0))
    soma.add_channel("leak", LeakChannel(g_max=0.3))
    soma.add_channel("ca_l", CaLChannel(g_max=0.3))

    ap_prox = Compartment(id=1, type=CompartmentType.APICAL_DEND,
                          length=200.0, diameter=3.0, parent_id=0,
                          x=0, y=200)
    ap_prox.add_channel("na",   NaChannel(g_max=30.0))
    ap_prox.add_channel("k",    KChannel(g_max=20.0))
    ap_prox.add_channel("leak", LeakChannel(g_max=0.03))
    ap_prox.add_channel("ka",   KAChannel(g_max=8.0))
    ap_prox.add_channel("hcn",  HCNChannel(g_max=0.1))

    ap_dist = Compartment(id=2, type=CompartmentType.APICAL_DEND,
                          length=300.0, diameter=2.0, parent_id=1,
                          x=0, y=500)
    ap_dist.add_channel("na",   NaChannel(g_max=20.0))
    ap_dist.add_channel("k",    KChannel(g_max=15.0))
    ap_dist.add_channel("leak", LeakChannel(g_max=0.03))
    ap_dist.add_channel("ca_l", CaLChannel(g_max=0.5))
    ap_dist.add_channel("ka",   KAChannel(g_max=15.0))

    basal = Compartment(id=3, type=CompartmentType.BASAL_DEND,
                        length=150.0, diameter=2.5, parent_id=0,
                        x=0, y=-150)
    basal.add_channel("na",   NaChannel(g_max=20.0))
    basal.add_channel("k",    KChannel(g_max=15.0))
    basal.add_channel("leak", LeakChannel(g_max=0.03))

    axon = Compartment(id=4, type=CompartmentType.AXON_HILLOCK,
                       length=30.0, diameter=1.0, parent_id=0,
                       x=0, y=-30)
    axon.add_channel("na",   NaChannel(g_max=5000.0))  # High density at AIS
    axon.add_channel("k",    KChannel(g_max=200.0))
    axon.add_channel("leak", LeakChannel(g_max=0.3))

    for comp in [soma, ap_prox, ap_dist, basal, axon]:
        neuron.add_compartment(comp)

    neuron.initialize(V_init=-65.0)
    return neuron


def build_parvalbumin_interneuron(neuron_id: int = 0) -> MultiCompartmentNeuron:
    """
    Fast-spiking parvalbumin interneuron — 2-compartment model.
    """
    neuron = MultiCompartmentNeuron(neuron_id)

    soma = Compartment(id=0, type=CompartmentType.SOMA,
                       length=15.0, diameter=15.0)
    soma.add_channel("na",   NaChannel(g_max=500.0))
    soma.add_channel("k",    KChannel(g_max=100.0))
    soma.add_channel("ka",   KAChannel(g_max=5.0))
    soma.add_channel("leak", LeakChannel(g_max=0.25))

    dend = Compartment(id=1, type=CompartmentType.BASAL_DEND,
                       length=100.0, diameter=1.5, parent_id=0)
    dend.add_channel("na",   NaChannel(g_max=50.0))
    dend.add_channel("k",    KChannel(g_max=30.0))
    dend.add_channel("leak", LeakChannel(g_max=0.05))

    neuron.add_compartment(soma)
    neuron.add_compartment(dend)
    neuron.initialize(V_init=-68.0)
    return neuron