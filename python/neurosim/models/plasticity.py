"""
neurosim/models/plasticity.py

Synaptic plasticity models:
- Spike-Timing Dependent Plasticity (STDP)
- Short-Term Plasticity (Tsodyks-Markram model)
- Long-Term Potentiation / Depression (BCM rule)
- Homeostatic synaptic scaling
- Metaplasticity
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum, auto


# ─────────────────────────────────────────────────────────────
#  STDP — Spike-Timing Dependent Plasticity
# ─────────────────────────────────────────────────────────────
@dataclass
class STDPRule:
    """
    Classical asymmetric STDP (Bi & Poo, 1998).

    dw = A_plus  * exp(-Δt / tau_plus)   if Δt > 0  (pre before post → LTP)
    dw = -A_minus * exp( Δt / tau_minus) if Δt < 0  (post before pre → LTD)

    Extended with:
    - Hard weight bounds [w_min, w_max]
    - Multiplicative scaling (van Rossum 2000)
    - Nearest-neighbor or all-to-all pairing
    """
    A_plus:     float = 0.01     # LTP amplitude
    A_minus:    float = 0.0105   # LTD amplitude  (slight asymmetry → net LTD)
    tau_plus:   float = 20.0     # ms  LTP time constant
    tau_minus:  float = 20.0     # ms  LTD time constant
    w_min:      float = 0.0      # minimum weight
    w_max:      float = 1.0      # maximum weight
    multiplicative: bool = True   # weight-dependent scaling

    def dw(self, delta_t: float, w: float) -> float:
        """
        Compute weight change given spike timing difference Δt = t_post - t_pre.
        Positive Δt → LTP, negative → LTD.
        """
        if delta_t >= 0:
            raw = self.A_plus * np.exp(-delta_t / self.tau_plus)
            if self.multiplicative:
                return raw * (self.w_max - w)
            return raw
        else:
            raw = -self.A_minus * np.exp(delta_t / self.tau_minus)
            if self.multiplicative:
                return raw * (w - self.w_min)
            return raw

    def update_weight(self, w: float, delta_t: float) -> float:
        new_w = w + self.dw(delta_t, w)
        return float(np.clip(new_w, self.w_min, self.w_max))


@dataclass
class STDPState:
    """Per-synapse trace variables for online STDP."""
    x_pre:   float = 0.0   # Pre-synaptic trace
    x_post:  float = 0.0   # Post-synaptic trace
    weight:  float = 0.5
    tau_pre: float = 20.0
    tau_post: float = 20.0

    def pre_spike(self, rule: STDPRule) -> None:
        """Called when pre-synaptic neuron fires."""
        # LTD: post trace drives weight down
        self.weight = rule.update_weight(self.weight, -self.x_post)
        self.x_pre  = 1.0  # reset pre trace

    def post_spike(self, rule: STDPRule) -> None:
        """Called when post-synaptic neuron fires."""
        # LTP: pre trace drives weight up
        self.weight  = rule.update_weight(self.weight, self.x_pre)
        self.x_post  = 1.0  # reset post trace

    def decay(self, dt: float) -> None:
        self.x_pre  -= self.x_pre  / self.tau_pre  * dt
        self.x_post -= self.x_post / self.tau_post * dt


# ─────────────────────────────────────────────────────────────
#  Short-Term Plasticity (Tsodyks-Markram)
# ─────────────────────────────────────────────────────────────
class STPType(Enum):
    FACILITATING  = auto()   # e.g., L5→L2 excitatory
    DEPRESSING    = auto()   # e.g., L4→L2 excitatory
    PSEUDO_LINEAR = auto()   # balanced


@dataclass
class TsodyksMarkramSynapse:
    """
    Tsodyks–Markram short-term plasticity model (1997).

    State variables:
    - u:  utilization (release probability)
    - x:  fraction of available resources
    - I:  synaptic current (filtered)

    Parameters for phenomenological synapse types:
    - Facilitating: low U, long tau_rec, long tau_fac
    - Depressing:   high U, short tau_rec, short tau_fac
    """
    U:        float          # baseline release probability [0,1]
    tau_rec:  float          # ms  recovery time constant
    tau_fac:  float          # ms  facilitation time constant (0 = none)
    tau_I:    float = 3.0    # ms  synaptic current decay
    A:        float = 1.0    # absolute synaptic efficacy

    # State
    u: float = field(init=False)
    x: float = field(default=1.0, init=False)
    I: float = field(default=0.0, init=False)
    last_t: float = field(default=-1e9, init=False)

    def __post_init__(self):
        self.u = self.U

    @classmethod
    def facilitating(cls) -> "TsodyksMarkramSynapse":
        return cls(U=0.15, tau_rec=130.0, tau_fac=670.0)

    @classmethod
    def depressing(cls) -> "TsodyksMarkramSynapse":
        return cls(U=0.45, tau_rec=20.0, tau_fac=0.0)

    @classmethod
    def pseudo_linear(cls) -> "TsodyksMarkramSynapse":
        return cls(U=0.29, tau_rec=190.0, tau_fac=125.0)

    def spike(self, t: float) -> float:
        """Process a pre-synaptic spike at time t. Return released fraction."""
        dt = t - self.last_t if self.last_t > -1e8 else 1e9
        self.last_t = t

        # Recovery between spikes
        self.x += (1.0 - self.x) * (1.0 - np.exp(-dt / self.tau_rec))

        # Facilitation
        if self.tau_fac > 0:
            self.u += (self.U - self.u) * (1.0 - np.exp(-dt / self.tau_fac))
            self.u += self.U * (1.0 - self.u)
        else:
            self.u = self.U

        # Released fraction
        released = self.u * self.x
        self.x  -= released
        self.I   = self.A * released
        return released

    def update(self, dt: float) -> float:
        """Decay synaptic current between spikes."""
        self.I *= np.exp(-dt / self.tau_I)
        return self.I


# ─────────────────────────────────────────────────────────────
#  BCM (Bienenstock-Cooper-Munro) plasticity rule
# ─────────────────────────────────────────────────────────────
@dataclass
class BCMRule:
    """
    BCM sliding threshold plasticity.
    dw/dt = phi(y, θ_M) * x
    phi = y * (y - θ_M)
    θ_M slides based on mean postsynaptic activity.

    Implements: Bienenstock, Cooper & Munro (1982)
    """
    tau_theta: float = 1000.0  # ms  threshold sliding time constant
    eta:       float = 0.001   # learning rate
    theta_M:   float = 1.0     # sliding modification threshold

    def phi(self, y: float) -> float:
        """BCM nonlinearity."""
        return y * (y - self.theta_M)

    def update(self, w: float, x: float, y: float, dt: float) -> Tuple[float, float]:
        """
        Update weight and threshold.
        w: current weight
        x: presynaptic activity
        y: postsynaptic activity
        Returns: (new_w, new_theta_M)
        """
        dw         = self.eta * self.phi(y) * x
        d_theta    = (y**2 - self.theta_M) / self.tau_theta
        new_w      = max(0.0, w + dw * dt)
        new_theta  = self.theta_M + d_theta * dt
        return new_w, max(0.1, new_theta)


# ─────────────────────────────────────────────────────────────
#  Homeostatic Synaptic Scaling
# ─────────────────────────────────────────────────────────────
@dataclass
class SynapticScaling:
    """
    Homeostatic plasticity via multiplicative synaptic scaling.
    Adjusts all synaptic weights to maintain target mean firing rate.
    Turrigiano & Nelson (2004).
    """
    target_rate: float = 5.0     # Hz  target mean firing rate
    tau_h:       float = 1e5     # ms  homeostatic time constant (~24h → 86400000ms)
    eta_h:       float = 1e-5    # learning rate

    def scale_weights(self, weights: np.ndarray,
                      mean_rate: float, dt: float) -> np.ndarray:
        """
        Multiplicatively scale all weights to drive rate toward target.
        """
        error  = self.target_rate - mean_rate
        factor = 1.0 + self.eta_h * error * dt
        return weights * np.clip(factor, 0.9, 1.1)


# ─────────────────────────────────────────────────────────────
#  Neurotransmitter Diffusion Model
# ─────────────────────────────────────────────────────────────
@dataclass
class NeurotransmitterDiffusion:
    """
    Simplified 2D Gaussian diffusion of neurotransmitter
    in the synaptic cleft and extrasynaptic space.

    ∂C/∂t = D ∇²C - k_clear * C + R(t)

    Where:
    - D: diffusion coefficient (μm²/ms)
    - k_clear: clearance rate (1/ms) — reuptake + degradation
    - R: release rate
    """
    D:        float = 0.4      # μm²/ms  (glutamate in aqueous: ~0.3-0.76)
    k_clear:  float = 0.5      # 1/ms
    cleft_w:  float = 20e-3    # μm  synaptic cleft width
    dt:       float = 0.025    # ms  time step

    # Grid for diffusion (N x N spatial points)
    N:        int   = 10
    dx:       float = 0.05     # μm  spatial resolution

    def __post_init__(self):
        self.C     = np.zeros((self.N, self.N))   # concentration grid (mM)
        self.alpha = self.D * self.dt / self.dx**2

        if self.alpha > 0.5:
            raise ValueError(
                f"Diffusion stability condition violated: α={self.alpha:.3f} > 0.5. "
                "Reduce dt or increase dx.")

    def release(self, amount: float, cx: int = None, cy: int = None) -> None:
        """Release neurotransmitter at cleft center."""
        cx = cx or self.N // 2
        cy = cy or self.N // 2
        self.C[cx, cy] += amount

    def step(self) -> np.ndarray:
        """Advance diffusion by one time step (finite difference)."""
        C = self.C
        # 2D discrete Laplacian with zero-flux boundary
        lap = (np.roll(C,  1, 0) + np.roll(C, -1, 0) +
               np.roll(C,  1, 1) + np.roll(C, -1, 1) - 4 * C)
        self.C = C + self.alpha * lap - self.k_clear * C * self.dt
        self.C = np.maximum(self.C, 0.0)
        return self.C

    def peak_concentration(self) -> float:
        return float(self.C.max())

    def total_transmitter(self) -> float:
        return float(self.C.sum() * self.dx**2)


# ─────────────────────────────────────────────────────────────
#  Glial cell (astrocyte) interactions
# ─────────────────────────────────────────────────────────────
@dataclass
class Astrocyte:
    """
    Simplified astrocyte tripartite synapse model.
    - Glutamate uptake (EAATs)
    - Gliotransmitter release (ATP, glutamate, d-serine)
    - IP3-dependent calcium waves
    - Potassium buffering
    """
    uptake_rate:   float = 2.0     # 1/ms  glutamate uptake rate constant
    ca_threshold:  float = 0.5     # μM    Ca2+ threshold for gliotransmission
    k_buffer:      float = 0.1     # K+ buffering coefficient
    ip3:           float = 0.0     # IP3 concentration (μM)
    ca:            float = 0.1     # intracellular Ca2+ (μM)
    glut_ext:      float = 0.0     # extracellular glutamate (mM)

    # IP3 receptor parameters
    d1:    float = 0.13   # dissociation constant IP3
    d5:    float = 0.082  # Ca2+ inhibition
    a2:    float = 0.2    # rate constant

    def uptake_glutamate(self, glut_syn: float, dt: float) -> float:
        """Remove glutamate from synaptic cleft."""
        uptaken       = self.uptake_rate * glut_syn * dt
        self.glut_ext += uptaken * 0.01  # small fraction enters astrocyte
        return max(0.0, glut_syn - uptaken)

    def update_calcium(self, dt: float) -> float:
        """
        Simplified Li–Rinzel IP3R calcium model.
        Returns astrocytic Ca2+ level.
        """
        # IP3 drives Ca2+ release from ER
        m_inf = self.ip3 / (self.ip3 + self.d1)
        h_inf = self.d5 / (self.ca + self.d5)
        j_release = m_inf ** 3 * h_inf ** 3 * (2.0 - self.ca)
        j_pump    = 0.5 * self.ca ** 2 / (0.3 ** 2 + self.ca ** 2)
        dca = (j_release - j_pump) * dt
        self.ca = max(0.05, self.ca + dca)
        return self.ca

    def gliotransmit(self) -> Optional[str]:
        """Release gliotransmitter if Ca2+ threshold exceeded."""
        if self.ca > self.ca_threshold:
            self.ca *= 0.7  # partial depletion
            return "d_serine"  # coagonist for NMDA receptors
        return None

    def buffer_potassium(self, k_ext: float) -> float:
        """Spatial K+ buffering (reduces extracellular [K+])."""
        return k_ext * (1.0 - self.k_buffer)
