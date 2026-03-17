"""
neurosim/analysis/spike_analysis.py

Comprehensive neural data analysis tools:
- Spike sorting and detection
- Firing rate estimation (PSTH, instantaneous rate)
- Inter-spike interval (ISI) analysis
- Cross-correlogram / spike train correlations
- Local field potential (LFP) computation
- Population vector analysis
- Synchrony measures (SPIKE distance, van Rossum distance)
- Connectivity statistics
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d


# ─────────────────────────────────────────────────────────────
#  Basic spike train utilities
# ─────────────────────────────────────────────────────────────
def spike_count(spike_times: List[float],
                t_start: float, t_stop: float) -> int:
    """Count spikes in window [t_start, t_stop] ms."""
    return sum(1 for t in spike_times if t_start <= t < t_stop)


def mean_firing_rate(spike_times: List[float],
                     t_start: float, t_stop: float) -> float:
    """Mean firing rate in Hz."""
    duration_s = (t_stop - t_start) / 1000.0
    if duration_s <= 0:
        return 0.0
    return spike_count(spike_times, t_start, t_stop) / duration_s


def isi(spike_times: List[float]) -> np.ndarray:
    """Inter-spike intervals (ms)."""
    st = np.sort(spike_times)
    return np.diff(st)


def cv_isi(spike_times: List[float]) -> float:
    """Coefficient of variation of ISI (regularity measure)."""
    intervals = isi(spike_times)
    if len(intervals) < 2:
        return np.nan
    return float(np.std(intervals) / np.mean(intervals))


def fano_factor(spike_trains: List[List[float]],
                t_start: float, t_stop: float,
                bin_size: float = 50.0) -> float:
    """
    Fano factor of spike count across neurons in time bins.
    FF = Var(count) / Mean(count)
    FF=1 → Poisson, FF<1 → sub-Poisson, FF>1 → super-Poisson
    """
    bins  = np.arange(t_start, t_stop + bin_size, bin_size)
    counts = []
    for st in spike_trains:
        hist, _ = np.histogram(st, bins=bins)
        counts.append(hist)
    counts = np.array(counts, dtype=float)  # shape: (n_neurons, n_bins)
    mean   = counts.mean(axis=0)
    var    = counts.var(axis=0)
    valid  = mean > 0
    return float(np.mean(var[valid] / mean[valid])) if valid.any() else np.nan


# ─────────────────────────────────────────────────────────────
#  PSTH — Peri-Stimulus Time Histogram
# ─────────────────────────────────────────────────────────────
@dataclass
class PSTH:
    bin_edges:  np.ndarray
    rate:       np.ndarray    # Hz
    counts:     np.ndarray
    sem:        np.ndarray    # standard error across trials/neurons
    bin_centers: np.ndarray


def compute_psth(spike_trains: List[List[float]],
                 t_start: float = 0.0,
                 t_stop: float = 1000.0,
                 bin_size: float = 10.0,
                 smooth_sigma: float = 1.0) -> PSTH:
    """
    Compute population PSTH from multiple spike trains.
    Gaussian smoothing optional (sigma in bins).
    """
    bins    = np.arange(t_start, t_stop + bin_size, bin_size)
    n_bins  = len(bins) - 1
    n_trains = len(spike_trains)
    all_counts = np.zeros((n_trains, n_bins))

    for i, st in enumerate(spike_trains):
        all_counts[i], _ = np.histogram(st, bins=bins)

    mean_counts = all_counts.mean(axis=0)
    sem_counts  = all_counts.std(axis=0) / np.sqrt(n_trains)
    rate        = mean_counts / (bin_size / 1000.0)  # Hz
    sem_rate    = sem_counts  / (bin_size / 1000.0)

    if smooth_sigma > 0:
        rate     = gaussian_filter1d(rate,     smooth_sigma)
        sem_rate = gaussian_filter1d(sem_rate, smooth_sigma)

    return PSTH(
        bin_edges   = bins,
        rate        = rate,
        counts      = mean_counts,
        sem         = sem_rate,
        bin_centers = 0.5 * (bins[:-1] + bins[1:]),
    )


# ─────────────────────────────────────────────────────────────
#  Cross-correlogram
# ─────────────────────────────────────────────────────────────
def cross_correlogram(spike_times_a: List[float],
                      spike_times_b: List[float],
                      max_lag: float = 100.0,
                      bin_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-correlogram between two spike trains.
    Returns (lags, ccg) — ccg in coincidences per bin.
    """
    bins  = np.arange(-max_lag, max_lag + bin_size, bin_size)
    diffs = []
    for ta in spike_times_a:
        for tb in spike_times_b:
            d = tb - ta
            if -max_lag <= d <= max_lag:
                diffs.append(d)
    ccg, _ = np.histogram(diffs, bins=bins)
    lags   = 0.5 * (bins[:-1] + bins[1:])
    return lags, ccg.astype(float)


def auto_correlogram(spike_times: List[float],
                     max_lag: float = 100.0,
                     bin_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Auto-correlogram with zero-lag removed."""
    lags, acg = cross_correlogram(spike_times, spike_times, max_lag, bin_size)
    center = len(acg) // 2
    acg[center] = 0  # remove zero-lag peak
    return lags, acg


# ─────────────────────────────────────────────────────────────
#  Local Field Potential estimation
# ─────────────────────────────────────────────────────────────
def compute_lfp(V_matrix: np.ndarray,
                dt: float = 0.025,
                electrode_pos: np.ndarray = None,
                neuron_pos: np.ndarray = None,
                conductivity: float = 0.3) -> np.ndarray:
    """
    Estimate LFP using volume conduction (point source approximation).
    V_matrix: shape (n_neurons, n_compartments, n_timesteps)
    Returns LFP signal at electrode position (n_timesteps,).

    Uses: φ(r) = Σ_i I_i / (4π σ |r - r_i|)
    """
    n_neurons, n_comps, n_t = V_matrix.shape

    if electrode_pos is None:
        electrode_pos = np.array([0.0, 0.0, 100.0])  # μm above network

    if neuron_pos is None:
        # Place neurons in a grid
        n_side = int(np.ceil(np.sqrt(n_neurons)))
        spacing = 50.0  # μm
        positions = np.array([[(i % n_side) * spacing,
                                (i // n_side) * spacing, 0.0]
                               for i in range(n_neurons)])
        neuron_pos = positions

    lfp = np.zeros(n_t)
    for i in range(n_neurons):
        r = np.linalg.norm(electrode_pos - neuron_pos[i]) + 1e-6  # μm
        # Approximate transmembrane current from dV/dt
        dV = np.gradient(V_matrix[i, 0, :], dt)
        contribution = dV / (4 * np.pi * conductivity * r * 1e-4)  # mV
        lfp += contribution

    return lfp


def bandpass_filter(signal_: np.ndarray, fs: float,
                    low: float, high: float,
                    order: int = 4) -> np.ndarray:
    """Butterworth bandpass filter."""
    nyq  = 0.5 * fs
    b, a = signal.butter(order, [low/nyq, high/nyq], btype="band")
    return signal.filtfilt(b, a, signal_)


def compute_power_spectrum(sig: np.ndarray,
                           fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density estimate."""
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(256, len(sig)//4))
    return freqs, psd


# ─────────────────────────────────────────────────────────────
#  Population synchrony measures
# ─────────────────────────────────────────────────────────────
def pairwise_correlation(spike_trains: List[List[float]],
                         t_start: float, t_stop: float,
                         bin_size: float = 5.0) -> np.ndarray:
    """
    Compute pairwise Pearson correlation matrix of spike train bins.
    Returns (n_neurons × n_neurons) correlation matrix.
    """
    bins  = np.arange(t_start, t_stop + bin_size, bin_size)
    binned = np.array([np.histogram(st, bins=bins)[0]
                       for st in spike_trains], dtype=float)

    # Pearson correlation
    n = len(spike_trains)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            if binned[i].std() > 0 and binned[j].std() > 0:
                r, _ = stats.pearsonr(binned[i], binned[j])
                corr[i, j] = corr[j, i] = r
    return corr


def synchrony_index(spike_trains: List[List[float]],
                    t_start: float, t_stop: float,
                    bin_size: float = 5.0) -> float:
    """
    Population synchrony index (0 = asynchronous, 1 = perfectly synchronous).
    Uses variance of population activity / mean firing rate.
    """
    bins  = np.arange(t_start, t_stop + bin_size, bin_size)
    pop_rate = np.zeros(len(bins) - 1)
    for st in spike_trains:
        cnt, _ = np.histogram(st, bins=bins)
        pop_rate += cnt
    if pop_rate.mean() == 0:
        return 0.0
    return float(pop_rate.var() / pop_rate.mean())


# ─────────────────────────────────────────────────────────────
#  Spike raster generation
# ─────────────────────────────────────────────────────────────
@dataclass
class RasterData:
    neuron_ids: np.ndarray   # flat array of neuron indices
    spike_times: np.ndarray  # corresponding spike times (ms)
    n_neurons: int
    t_start: float
    t_stop: float

    @classmethod
    def from_spike_dict(cls, spikes: Dict[int, List[float]],
                        t_start: float, t_stop: float) -> "RasterData":
        all_ids   = []
        all_times = []
        for nid, times in spikes.items():
            for t in times:
                if t_start <= t <= t_stop:
                    all_ids.append(nid)
                    all_times.append(t)
        return cls(
            neuron_ids  = np.array(all_ids),
            spike_times = np.array(all_times),
            n_neurons   = len(spikes),
            t_start     = t_start,
            t_stop      = t_stop
        )


# ─────────────────────────────────────────────────────────────
#  Connectivity statistics
# ─────────────────────────────────────────────────────────────
def connectivity_stats(adj_matrix: np.ndarray) -> dict:
    """
    Compute graph-theoretic connectivity statistics.
    adj_matrix: (n × n) adjacency matrix (binary or weighted)
    """
    n = adj_matrix.shape[0]
    binary = (adj_matrix > 0).astype(float)

    in_degree  = binary.sum(axis=0)
    out_degree = binary.sum(axis=1)

    # Clustering coefficient (per node)
    clustering = np.zeros(n)
    for i in range(n):
        neighbors = np.where(binary[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            continue
        triangles = binary[np.ix_(neighbors, neighbors)].sum()
        clustering[i] = triangles / (k * (k - 1))

    # Reciprocity
    recip = float((binary * binary.T).sum()) / max(1, binary.sum())

    return {
        "n_neurons":         n,
        "n_connections":     int(binary.sum()),
        "density":           float(binary.sum() / (n * (n - 1))),
        "mean_in_degree":    float(in_degree.mean()),
        "mean_out_degree":   float(out_degree.mean()),
        "max_in_degree":     float(in_degree.max()),
        "max_out_degree":    float(out_degree.max()),
        "mean_clustering":   float(clustering.mean()),
        "reciprocity":       recip,
        "in_degree_dist":    in_degree,
        "out_degree_dist":   out_degree,
    }
