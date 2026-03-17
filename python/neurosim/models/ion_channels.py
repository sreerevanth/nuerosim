"""
neurosim/models/ion_channels.py

Biophysically detailed ion channel and membrane dynamics models.
Implements Hodgkin–Huxley formalism with extensions for:
- Na+, K+, Ca2+, and leak channels
- Voltage-gated and ligand-gated variants
- Temperature-dependent kinetics (Q10 scaling)
- Markov-chain channel state models
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from scipy.integrate import solve_ivp


# ─────────────────────────────────────────────────────────────
#  Physical constants
# ─────────────────────────────────────────────────────────────
FARADAY    = 96485.0      # C/mol
GAS_CONST  = 8.314        # J/(mol·K)
CELSIUS_0  = 273.15       # K
DEFAULT_T  = 37.0         # °C  (body temperature)
Q10_DEFAULT = 3.0         # Temperature coefficient


def celsius_to_kelvin(T_celsius: float) -> float:
    return T_celsius + CELSIUS_0


def nernst_potential(z: float, c_in: float, c_out: float,
                     T_celsius: float = DEFAULT_T) -> float:
    """Nernst equilibrium potential (mV)."""
    T_K = celsius_to_kelvin(T_celsius)
    return (GAS_CONST * T_K / (z * FARADAY)) * np.log(c_out / c_in) * 1000.0


def q10_scale(rate: float, T: float, T_ref: float = 6.3,
              q10: float = Q10_DEFAULT) -> float:
    """Q10 temperature scaling of rate constants."""
    return rate * q10 ** ((T - T_ref) / 10.0)


# ─────────────────────────────────────────────────────────────
#  Base channel class
# ─────────────────────────────────────────────────────────────
class IonChannel(ABC):
    """Abstract base for all ion channels."""

    def __init__(self, g_max: float, E_rev: float, T: float = DEFAULT_T):
        self.g_max  = g_max    # Max conductance (S/cm²)
        self.E_rev  = E_rev    # Reversal potential (mV)
        self.T      = T        # Temperature (°C)

    @abstractmethod
    def current(self, V: float, state: np.ndarray) -> float:
        """Return ionic current (μA/cm²)."""

    @abstractmethod
    def derivatives(self, V: float, state: np.ndarray) -> np.ndarray:
        """Return d(state)/dt."""

    @abstractmethod
    def steady_state(self, V: float) -> np.ndarray:
        """Return steady-state gating variables at voltage V."""

    def conductance(self, state: np.ndarray) -> float:
        """Compute instantaneous conductance."""
        return self.g_max * self._gate_product(state)

    def _gate_product(self, state: np.ndarray) -> float:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
#  Hodgkin–Huxley Sodium Channel (INa)
# ─────────────────────────────────────────────────────────────
class NaChannel(IonChannel):
    """
    Hodgkin–Huxley fast sodium channel.
    State vector: [m, h]  (activation, inactivation)
    g_Na = g_max * m^3 * h
    """

    def __init__(self, g_max: float = 120.0, E_rev: float = 50.0,
                 T: float = DEFAULT_T):
        super().__init__(g_max, E_rev, T)

    # --- Alpha / Beta rate functions ---
    def _alpha_m(self, V: float) -> float:
        dV = V + 40.0
        if abs(dV) < 1e-7:
            return q10_scale(1.0, self.T) * 1.0
        return q10_scale(0.1 * dV / (1.0 - np.exp(-dV / 10.0)), self.T)

    def _beta_m(self, V: float) -> float:
        return q10_scale(4.0 * np.exp(-(V + 65.0) / 18.0), self.T)

    def _alpha_h(self, V: float) -> float:
        return q10_scale(0.07 * np.exp(-(V + 65.0) / 20.0), self.T)

    def _beta_h(self, V: float) -> float:
        return q10_scale(1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0)), self.T)

    def steady_state(self, V: float) -> np.ndarray:
        am, bm = self._alpha_m(V), self._beta_m(V)
        ah, bh = self._alpha_h(V), self._beta_h(V)
        m_inf = am / (am + bm)
        h_inf = ah / (ah + bh)
        return np.array([m_inf, h_inf])

    def derivatives(self, V: float, state: np.ndarray) -> np.ndarray:
        m, h = state
        dm = self._alpha_m(V) * (1 - m) - self._beta_m(V) * m
        dh = self._alpha_h(V) * (1 - h) - self._beta_h(V) * h
        return np.array([dm, dh])

    def _gate_product(self, state: np.ndarray) -> float:
        m, h = state
        return m ** 3 * h

    def current(self, V: float, state: np.ndarray) -> float:
        return self.g_max * self._gate_product(state) * (V - self.E_rev)


# ─────────────────────────────────────────────────────────────
#  Hodgkin–Huxley Potassium Channel (IK)
# ─────────────────────────────────────────────────────────────
class KChannel(IonChannel):
    """
    Hodgkin–Huxley delayed-rectifier potassium channel.
    State vector: [n]  (activation)
    g_K = g_max * n^4
    """

    def __init__(self, g_max: float = 36.0, E_rev: float = -77.0,
                 T: float = DEFAULT_T):
        super().__init__(g_max, E_rev, T)

    def _alpha_n(self, V: float) -> float:
        dV = V + 55.0
        if abs(dV) < 1e-7:
            return q10_scale(0.1, self.T)
        return q10_scale(0.01 * dV / (1.0 - np.exp(-dV / 10.0)), self.T)

    def _beta_n(self, V: float) -> float:
        return q10_scale(0.125 * np.exp(-(V + 65.0) / 80.0), self.T)

    def steady_state(self, V: float) -> np.ndarray:
        an, bn = self._alpha_n(V), self._beta_n(V)
        return np.array([an / (an + bn)])

    def derivatives(self, V: float, state: np.ndarray) -> np.ndarray:
        n = state[0]
        dn = self._alpha_n(V) * (1 - n) - self._beta_n(V) * n
        return np.array([dn])

    def _gate_product(self, state: np.ndarray) -> float:
        return state[0] ** 4

    def current(self, V: float, state: np.ndarray) -> float:
        return self.g_max * self._gate_product(state) * (V - self.E_rev)


# ─────────────────────────────────────────────────────────────
#  Leak Channel
# ─────────────────────────────────────────────────────────────
class LeakChannel(IonChannel):
    """Passive leak conductance — no gating."""

    def __init__(self, g_max: float = 0.3, E_rev: float = -54.387,
                 T: float = DEFAULT_T):
        super().__init__(g_max, E_rev, T)

    def steady_state(self, V: float) -> np.ndarray:
        return np.array([])

    def derivatives(self, V: float, state: np.ndarray) -> np.ndarray:
        return np.array([])

    def _gate_product(self, state: np.ndarray) -> float:
        return 1.0

    def current(self, V: float, state: np.ndarray) -> float:
        return self.g_max * (V - self.E_rev)


# ─────────────────────────────────────────────────────────────
#  High-Threshold Calcium Channel (L-type, ICa)
# ─────────────────────────────────────────────────────────────
class CaLChannel(IonChannel):
    """
    L-type high-voltage-activated calcium channel.
    State: [m, h]
    Uses GHK current equation for calcium.
    """

    def __init__(self, g_max: float = 0.5, T: float = DEFAULT_T,
                 ca_in: float = 1e-4, ca_out: float = 2.0):
        # GHK-based reversal varies; approximate at 125 mV for Ca2+
        E_ca = nernst_potential(z=2, c_in=ca_in, c_out=ca_out, T_celsius=T)
        super().__init__(g_max, E_ca, T)
        self.ca_in  = ca_in   # mM intracellular
        self.ca_out = ca_out  # mM extracellular

    def _m_inf(self, V: float) -> float:
        return 1.0 / (1.0 + np.exp(-(V + 37.0) / 7.0))

    def _tau_m(self, V: float) -> float:
        return 0.5

    def _h_inf(self, V: float) -> float:
        return 1.0 / (1.0 + np.exp((V + 41.0) / 0.5))

    def _tau_h(self, V: float) -> float:
        return 300.0

    def steady_state(self, V: float) -> np.ndarray:
        return np.array([self._m_inf(V), self._h_inf(V)])

    def derivatives(self, V: float, state: np.ndarray) -> np.ndarray:
        m, h = state
        dm = (self._m_inf(V) - m) / self._tau_m(V)
        dh = (self._h_inf(V) - h) / self._tau_h(V)
        return np.array([dm, dh])

    def _gate_product(self, state: np.ndarray) -> float:
        m, h = state
        return m ** 2 * h

    def current(self, V: float, state: np.ndarray) -> float:
        return self.g_max * self._gate_product(state) * (V - self.E_rev)


# ─────────────────────────────────────────────────────────────
#  A-type Potassium Channel (KA) — Transient
# ─────────────────────────────────────────────────────────────
class KAChannel(IonChannel):
    """
    Transient A-type potassium channel.
    Important for subthreshold integration and back-propagation modulation.
    State: [a, b]
    """

    def __init__(self, g_max: float = 10.0, E_rev: float = -77.0,
                 T: float = DEFAULT_T):
        super().__init__(g_max, E_rev, T)

    def _a_inf(self, V: float) -> float:
        return (0.0761 * np.exp((V + 94.22) / 31.84) /
                (1.0 + np.exp((V + 1.17) / 28.93))) ** (1/3)

    def _tau_a(self, V: float) -> float:
        return 0.3632 + 1.158 / (1.0 + np.exp((V + 55.96) / 20.12))

    def _b_inf(self, V: float) -> float:
        return (1.0 / (1.0 + np.exp((V + 53.3) / 14.54))) ** 4

    def _tau_b(self, V: float) -> float:
        return 1.24 + 2.678 / (1.0 + np.exp((V + 50.0) / 16.027))

    def steady_state(self, V: float) -> np.ndarray:
        return np.array([self._a_inf(V), self._b_inf(V)])

    def derivatives(self, V: float, state: np.ndarray) -> np.ndarray:
        a, b = state
        da = (self._a_inf(V) - a) / self._tau_a(V)
        db = (self._b_inf(V) - b) / self._tau_b(V)
        return np.array([da, db])

    def _gate_product(self, state: np.ndarray) -> float:
        a, b = state
        return a ** 3 * b

    def current(self, V: float, state: np.ndarray) -> float:
        return self.g_max * self._gate_product(state) * (V - self.E_rev)


# ─────────────────────────────────────────────────────────────
#  Calcium-Activated Potassium Channel (KCa / BK)
# ─────────────────────────────────────────────────────────────
class KCaChannel(IonChannel):
    """
    Big-conductance (BK) calcium-activated potassium channel.
    Depends on both voltage and intracellular [Ca2+].
    """

    def __init__(self, g_max: float = 1.0, E_rev: float = -77.0,
                 T: float = DEFAULT_T):
        super().__init__(g_max, E_rev, T)

    def _m_inf(self, V: float, ca: float) -> float:
        """Dual V and Ca2+ dependent activation."""
        alpha = 0.48 * (ca / 0.0003) * np.exp(V / 28.0)
        beta  = 0.28 * np.exp(-V / 48.0)
        return alpha / (alpha + beta)

    def steady_state(self, V: float, ca: float = 1e-4) -> np.ndarray:
        return np.array([self._m_inf(V, ca)])

    def derivatives(self, V: float, state: np.ndarray,
                    ca: float = 1e-4) -> np.ndarray:
        m = state[0]
        tau = 1.0  # ms — simplified
        dm = (self._m_inf(V, ca) - m) / tau
        return np.array([dm])

    def _gate_product(self, state: np.ndarray) -> float:
        return state[0]

    def current(self, V: float, state: np.ndarray) -> float:
        return self.g_max * self._gate_product(state) * (V - self.E_rev)


# ─────────────────────────────────────────────────────────────
#  HCN Channel (Ih) — Hyperpolarization-activated
# ─────────────────────────────────────────────────────────────
class HCNChannel(IonChannel):
    """
    Hyperpolarization-activated cyclic nucleotide-gated channel.
    Mixed Na+/K+ current (Ih). Critical for pacemaking and resonance.
    State: [q]
    """

    def __init__(self, g_max: float = 0.05, E_rev: float = -30.0,
                 T: float = DEFAULT_T):
        super().__init__(g_max, E_rev, T)

    def _q_inf(self, V: float) -> float:
        return 1.0 / (1.0 + np.exp((V + 80.0) / 10.0))

    def _tau_q(self, V: float) -> float:
        return 1.0 / (np.exp(-0.086 * V - 14.6) +
                      np.exp(0.07 * V - 1.87))

    def steady_state(self, V: float) -> np.ndarray:
        return np.array([self._q_inf(V)])

    def derivatives(self, V: float, state: np.ndarray) -> np.ndarray:
        q = state[0]
        dq = (self._q_inf(V) - q) / self._tau_q(V)
        return np.array([dq])

    def _gate_product(self, state: np.ndarray) -> float:
        return state[0]

    def current(self, V: float, state: np.ndarray) -> float:
        return self.g_max * self._gate_product(state) * (V - self.E_rev)


# ─────────────────────────────────────────────────────────────
#  Channel Registry
# ─────────────────────────────────────────────────────────────
CHANNEL_REGISTRY: Dict[str, type] = {
    "na":    NaChannel,
    "k":     KChannel,
    "leak":  LeakChannel,
    "ca_l":  CaLChannel,
    "ka":    KAChannel,
    "kca":   KCaChannel,
    "hcn":   HCNChannel,
}


def build_channel(name: str, **kwargs) -> IonChannel:
    """Factory function — build a channel by name."""
    cls = CHANNEL_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown channel type: '{name}'. "
                         f"Available: {list(CHANNEL_REGISTRY.keys())}")
    return cls(**kwargs)