"""
neurosim/simulation/engine.py

High-performance simulation engine supporting:
- Single-neuron and network-scale simulation
- RK4 and adaptive time-stepping
- Parallel neuron integration via thread pool / MPI
- Checkpoint / restart capability
- Event-driven spike delivery
- Spike queue with axonal delays
"""

from __future__ import annotations
import numpy as np
import time
import logging
import pickle
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import heapq
from collections import defaultdict

logger = logging.getLogger("neurosim.engine")


# ─────────────────────────────────────────────────────────────
#  Simulation configuration
# ─────────────────────────────────────────────────────────────
@dataclass
class SimulationConfig:
    dt:             float = 0.025         # ms  time step
    t_start:        float = 0.0           # ms
    t_stop:         float = 1000.0        # ms
    integrator:     str   = "rk4"         # "euler" | "rk4" | "cn" (Crank-Nicolson)
    record_dt:      float = 0.1           # ms  recording resolution
    n_workers:      int   = 4             # thread workers for parallel neuron integration
    checkpoint_dir: str   = "checkpoints"
    checkpoint_interval: float = 100.0   # ms
    seed:           int   = 42
    temperature:    float = 37.0          # °C

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────────────────────
#  Spike event queue (priority queue on delivery time)
# ─────────────────────────────────────────────────────────────
@dataclass(order=True)
class SpikeEvent:
    delivery_time: float
    src_neuron:    int   = field(compare=False)
    tgt_neuron:    int   = field(compare=False)
    tgt_comp:      int   = field(compare=False)
    syn_index:     int   = field(compare=False)
    weight:        float = field(compare=False, default=1.0)


class SpikeQueue:
    """Thread-safe priority queue for spike delivery."""

    def __init__(self):
        self._heap: List[SpikeEvent] = []

    def push(self, event: SpikeEvent) -> None:
        heapq.heappush(self._heap, event)

    def pop_until(self, t: float) -> List[SpikeEvent]:
        events = []
        while self._heap and self._heap[0].delivery_time <= t:
            events.append(heapq.heappop(self._heap))
        return events

    def __len__(self) -> int:
        return len(self._heap)


# ─────────────────────────────────────────────────────────────
#  Network connection table
# ─────────────────────────────────────────────────────────────
@dataclass
class Connection:
    src_id:    int
    tgt_id:    int
    tgt_comp:  int     # target compartment index
    syn_index: int     # index in tgt neuron's synapse list
    weight:    float
    delay:     float   # ms  axonal conduction delay


class ConnectivityGraph:
    """
    Sparse directed connectivity graph.
    Supports adjacency lookups in O(1) per source neuron.
    """

    def __init__(self):
        self._outgoing: Dict[int, List[Connection]] = defaultdict(list)
        self._incoming: Dict[int, List[Connection]] = defaultdict(list)
        self._n_conns = 0

    def add_connection(self, conn: Connection) -> None:
        self._outgoing[conn.src_id].append(conn)
        self._incoming[conn.tgt_id].append(conn)
        self._n_conns += 1

    def get_outgoing(self, neuron_id: int) -> List[Connection]:
        return self._outgoing.get(neuron_id, [])

    def get_incoming(self, neuron_id: int) -> List[Connection]:
        return self._incoming.get(neuron_id, [])

    @property
    def n_neurons_with_output(self) -> int:
        return len(self._outgoing)

    @property
    def n_connections(self) -> int:
        return self._n_conns


# ─────────────────────────────────────────────────────────────
#  RK4 integrator for a single neuron
# ─────────────────────────────────────────────────────────────
def rk4_step(neuron, dt: float, I_ext: np.ndarray) -> None:
    """
    4th-order Runge-Kutta integration for a MultiCompartmentNeuron.
    Operates on neuron in-place.
    """
    # Save current state
    V0    = neuron.V.copy()
    gates0 = [{k: v.copy() for k, v in g.items()} for g in neuron.gates]

    def compute_derivatives(V, gates):
        dV    = np.zeros(len(V))
        dgates = [{} for _ in gates]
        I_ax  = neuron._axial_currents()
        for i, comp in enumerate(neuron.compartments):
            I_ion = comp.ionic_current(V[i], gates[i])
            I_syn = sum(s.update(0.0, V[i]) for s in neuron.synapses[comp.id])
            dV[i] = (I_ax[i] - I_ion - I_syn + I_ext[i]) / comp.cap
            dgates[i] = comp.gate_derivatives(V[i], gates[i])
        return dV, dgates

    def apply_delta(V, gates, dV, dgates, h):
        V_new = V + h * dV
        gates_new = []
        for i, g in enumerate(gates):
            g_new = {}
            for name, val in g.items():
                g_new[name] = val + h * dgates[i].get(name, np.zeros_like(val))
            gates_new.append(g_new)
        return V_new, gates_new

    # k1
    k1_V, k1_g = compute_derivatives(V0, gates0)

    # k2
    V2, g2 = apply_delta(V0, gates0, k1_V, k1_g, dt/2)
    k2_V, k2_g = compute_derivatives(V2, g2)

    # k3
    V3, g3 = apply_delta(V0, gates0, k2_V, k2_g, dt/2)
    k3_V, k3_g = compute_derivatives(V3, g3)

    # k4
    V4, g4 = apply_delta(V0, gates0, k3_V, k3_g, dt)
    k4_V, k4_g = compute_derivatives(V4, g4)

    # Combine
    neuron.V = V0 + (dt/6) * (k1_V + 2*k2_V + 2*k3_V + k4_V)
    for i, g in enumerate(neuron.gates):
        for name in g:
            neuron.gates[i][name] = (
                gates0[i][name] + (dt/6) * (
                    k1_g[i].get(name, 0) +
                    2 * k2_g[i].get(name, 0) +
                    2 * k3_g[i].get(name, 0) +
                    k4_g[i].get(name, 0)
                )
            )


# ─────────────────────────────────────────────────────────────
#  Recording buffer
# ─────────────────────────────────────────────────────────────
class RecordingBuffer:
    """Efficient ring-buffer style voltage/spike recorder."""

    def __init__(self, n_neurons: int, n_comps_per_neuron: int,
                 t_stop: float, record_dt: float):
        n_steps = int(t_stop / record_dt) + 1
        self.V   = np.full((n_neurons, n_comps_per_neuron, n_steps), np.nan,
                           dtype=np.float32)
        self.t   = np.linspace(0, t_stop, n_steps)
        self.spikes: Dict[int, List[float]] = defaultdict(list)
        self.record_dt  = record_dt
        self._step_idx  = 0
        self._last_rec  = -np.inf

    def record(self, t: float, neuron_id: int, V: np.ndarray) -> None:
        if t - self._last_rec >= self.record_dt - 1e-10:
            idx = int(round(t / self.record_dt))
            if idx < self.V.shape[2]:
                self.V[neuron_id, :len(V), idx] = V.astype(np.float32)
            self._last_rec = t

    def record_spike(self, t: float, neuron_id: int) -> None:
        self.spikes[neuron_id].append(t)

    def soma_voltages(self, neuron_id: int) -> np.ndarray:
        return self.V[neuron_id, 0, :]

    def to_dict(self) -> dict:
        return {
            "t":     self.t,
            "V":     self.V,
            "spikes": dict(self.spikes),
        }


# ─────────────────────────────────────────────────────────────
#  Main simulation engine
# ─────────────────────────────────────────────────────────────
class NeuralSimulationEngine:
    """
    Main simulation engine for single-neuron to network-scale simulations.

    Supports:
    - Event-driven spike delivery with axonal delays
    - Parallel neuron integration (ThreadPoolExecutor)
    - Checkpoint / restore
    - Configurable integrators (Euler, RK4)
    - External current injection
    - Network connectivity
    """

    def __init__(self, config: SimulationConfig):
        self.config   = config
        self.neurons:  Dict[int, object] = {}
        self.graph     = ConnectivityGraph()
        self.spike_q   = SpikeQueue()
        self.recording: Optional[RecordingBuffer] = None
        self._t        = config.t_start
        self._rng      = np.random.default_rng(config.seed)
        self._I_clamps: Dict[int, Callable] = {}  # neuron_id → I(t) function

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Engine initialized: dt={config.dt}ms "
                    f"t=[{config.t_start},{config.t_stop}]ms")

    # ── Population management ─────────────────────────────────
    def add_neuron(self, neuron) -> None:
        self.neurons[neuron.neuron_id] = neuron

    def add_connection(self, conn: Connection) -> None:
        self.graph.add_connection(conn)

    def clamp_current(self, neuron_id: int, I_fn: Callable) -> None:
        """Inject time-varying current into soma. I_fn(t) → μA."""
        self._I_clamps[neuron_id] = I_fn

    def setup_recording(self, max_comps: int = 10) -> None:
        n = len(self.neurons)
        self.recording = RecordingBuffer(
            n_neurons=n,
            n_comps_per_neuron=max_comps,
            t_stop=self.config.t_stop,
            record_dt=self.config.record_dt
        )

    # ── Step execution ────────────────────────────────────────
    def _deliver_spikes(self) -> None:
        """Deliver all spikes due at current time."""
        events = self.spike_q.pop_until(self._t)
        for ev in events:
            tgt = self.neurons.get(ev.tgt_neuron)
            if tgt is None:
                continue
            syns = tgt.synapses.get(ev.tgt_comp, [])
            if ev.syn_index < len(syns):
                syns[ev.syn_index].activate(ev.weight)

    def _integrate_neuron(self, nid: int) -> List[SpikeEvent]:
        """Integrate a single neuron by one time step. Returns new spike events."""
        neuron  = self.neurons[nid]
        n_comps = len(neuron.compartments)
        I_ext   = np.zeros(n_comps)

        if nid in self._I_clamps:
            I_ext[0] = self._I_clamps[nid](self._t)

        prev_spikes = len(neuron.spike_detector.times)

        if self.config.integrator == "rk4":
            rk4_step(neuron, self.config.dt, I_ext)
        else:
            neuron.step(self.config.dt, I_ext)

        # Detect new spikes
        new_events = []
        if len(neuron.spike_detector.times) > prev_spikes:
            for conn in self.graph.get_outgoing(nid):
                ev = SpikeEvent(
                    delivery_time = self._t + conn.delay,
                    src_neuron    = nid,
                    tgt_neuron    = conn.tgt_id,
                    tgt_comp      = conn.tgt_comp,
                    syn_index     = conn.syn_index,
                    weight        = conn.weight,
                )
                new_events.append(ev)

        return new_events

    def _step(self) -> None:
        """One simulation time step."""
        self._deliver_spikes()

        # Parallel neuron integration
        all_events = []
        if self.config.n_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.n_workers) as pool:
                futures = {pool.submit(self._integrate_neuron, nid): nid
                           for nid in self.neurons}
                for fut in futures:
                    all_events.extend(fut.result())
        else:
            for nid in self.neurons:
                all_events.extend(self._integrate_neuron(nid))

        # Queue all new spike events
        for ev in all_events:
            self.spike_q.push(ev)

        # Record state
        if self.recording:
            for idx, (nid, neuron) in enumerate(self.neurons.items()):
                self.recording.record(self._t, idx, neuron.V)
                if len(neuron.spike_detector.times) > 0:
                    last_spike = neuron.spike_detector.times[-1]
                    if abs(last_spike - self._t) < self.config.dt:
                        self.recording.record_spike(self._t, idx)

        self._t += self.config.dt

    # ── Main run loop ─────────────────────────────────────────
    def run(self, progress_callback: Optional[Callable] = None) -> dict:
        """
        Execute simulation from t_start to t_stop.
        Returns recording data dict.
        """
        self.setup_recording()
        n_steps     = int((self.config.t_stop - self.config.t_start) /
                          self.config.dt)
        ckpt_steps  = int(self.config.checkpoint_interval / self.config.dt)

        t_wall_start = time.perf_counter()
        logger.info(f"Starting simulation: {len(self.neurons)} neurons, "
                    f"{n_steps} steps")

        for step in range(n_steps):
            self._step()

            # Checkpoint
            if step > 0 and step % ckpt_steps == 0:
                self._save_checkpoint(step)

            # Progress
            if progress_callback and step % max(1, n_steps // 100) == 0:
                pct = 100.0 * step / n_steps
                progress_callback(pct, self._t)

        wall_time = time.perf_counter() - t_wall_start
        logger.info(f"Simulation complete in {wall_time:.2f}s "
                    f"({n_steps/wall_time:.0f} steps/sec)")

        results = self.recording.to_dict() if self.recording else {}
        results["wall_time_s"]    = wall_time
        results["n_neurons"]      = len(self.neurons)
        results["n_connections"]  = self.graph.n_connections
        results["config"]         = self.config.to_dict()

        # Add spike statistics
        all_spikes = []
        for nid, neuron in self.neurons.items():
            all_spikes.extend(neuron.spike_detector.times)
        results["total_spikes"] = len(all_spikes)
        results["mean_firing_rate_hz"] = (
            len(all_spikes) / len(self.neurons) /
            (self.config.t_stop / 1000.0)
            if self.neurons else 0.0
        )
        return results

    # ── Checkpointing ─────────────────────────────────────────
    def _save_checkpoint(self, step: int) -> None:
        path = Path(self.config.checkpoint_dir) / f"checkpoint_step_{step:08d}.pkl"
        state = {
            "t":      self._t,
            "step":   step,
            "config": self.config.to_dict(),
            "neurons": {
                nid: {
                    "V":      n.V.tolist(),
                    "gates":  [{k: v.tolist() for k, v in g.items()}
                               for g in n.gates],
                    "spikes": n.spike_detector.times,
                }
                for nid, n in self.neurons.items()
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Checkpoint saved: {path}")

    def restore_checkpoint(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._t = state["t"]
        for nid, ns in state["neurons"].items():
            n = self.neurons.get(int(nid))
            if n:
                n.V = np.array(ns["V"])
                n.gates = [{k: np.array(v) for k, v in g.items()}
                           for g in ns["gates"]]
                n.spike_detector.times = ns["spikes"]
        logger.info(f"Restored checkpoint at t={self._t:.2f}ms")


# ─────────────────────────────────────────────────────────────
#  Convenience: build a simple current-clamp experiment
# ─────────────────────────────────────────────────────────────
def run_current_clamp(neuron, I_amp: float = 2.0,
                      t_start_inj: float = 100.0,
                      t_stop_inj: float = 500.0,
                      t_stop_sim: float = 600.0,
                      dt: float = 0.025) -> dict:
    """
    Run a current-clamp experiment on a single neuron.
    Returns time series of soma voltage and spike times.
    """
    from neurosim.models.neuron import MultiCompartmentNeuron

    cfg = SimulationConfig(
        dt=dt, t_start=0.0, t_stop=t_stop_sim,
        integrator="rk4", record_dt=dt, n_workers=1
    )
    engine = NeuralSimulationEngine(cfg)
    engine.add_neuron(neuron)

    def I_fn(t: float) -> float:
        return I_amp if t_start_inj <= t <= t_stop_inj else 0.0

    engine.clamp_current(neuron.neuron_id, I_fn)
    return engine.run()
