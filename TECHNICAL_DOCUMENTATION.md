# NeuroSim Platform — Technical Documentation

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Biological Models](#2-biological-models)
3. [Simulation Engine](#3-simulation-engine)
4. [HPC & Parallelization](#4-hpc--parallelization)
5. [Data Architecture](#5-data-architecture)
6. [API Reference](#6-api-reference)
7. [Visualization Pipeline](#7-visualization-pipeline)
8. [ML Integration](#8-ml-integration)
9. [Research Workflows](#9-research-workflows)
10. [Developer Guide](#10-developer-guide)

---

## 1. System Architecture

### Overview

NeuroSim is a layered, microservice-based platform for multi-scale neural simulation.

```
┌──────────────────────────────────────────────────────────────┐
│                    Client Layer                              │
│    CLI  ·  REST API  ·  Dash Dashboard  ·  Jupyter           │
├───────────────┬──────────────────┬───────────────────────────┤
│  API Gateway  │  Job Scheduler   │  Visualization Service    │
│  (FastAPI)    │  (Celery/Redis)  │  (Dash/Plotly)            │
├───────────────┴──────────────────┴───────────────────────────┤
│                Simulation Core                               │
│  Python orchestration  ·  C++/CUDA kernels  ·  MPI layer     │
├──────────────┬───────────────────────────────────────────────┤
│  Data Layer  │  HDF5 · PostgreSQL · Redis · MinIO (S3)       │
└──────────────┴───────────────────────────────────────────────┘
```

### Component responsibilities

| Component                | Role |
|--------------------------|------|
| `neurosim.models`        | Biophysical neuron and channel models |
| `neurosim.simulation`    | Integration engine, spike queue, checkpointing |
| `neurosim.reconstruction`| Morphology ingestion, SWC, SONATA |
| `neurosim.analysis`      | Spike analysis, LFP, connectivity stats |
| `neurosim.ml`            | CMA-ES, surrogate models, pattern discovery |
| `neurosim.visualization` | Plotly figures, Dash dashboard |
| `neurosim.worker`        | Celery async task execution |
| `services/gateway`       | FastAPI REST service |
| `cpp/kernels`            | CUDA HH integration kernels |
| `cpp/mpi`               | MPI distributed simulation |

---

## 2. Biological Models

### 2.1 Ion Channel Formalism

All channels follow the Hodgkin–Huxley conductance-based formalism:

```
I_ion = g_max · gates · (V - E_rev)
```

Gate variables satisfy first-order kinetics:
```
dm/dt = α_m(V)·(1 - m) - β_m(V)·m
```

Rate functions are Q10-temperature-corrected:
```
α(T) = α_ref · Q10^((T - T_ref) / 10)
```

### 2.2 Channel Inventory

| Channel | Gene     | Region         | g_max (S/cm²) | Notes |
|---------|----------|----------------|----------------|-------|
| NaV     | SCN1A    | Soma, AIS      | 0.12–5.0       | Inactivating |
| Kdr     | KCNB1    | Soma, dend.    | 0.036          | Delayed rectifier |
| KA      | KCNA4    | Distal dend.   | 0.01–0.015     | A-type, transient |
| KCa(BK) | KCNMA1   | Soma           | 0.001          | Ca²⁺-activated |
| HCN     | HCN1/2   | Apical dend.   | 0.00005        | Mixed Na⁺/K⁺ |
| CaL     | CACNA1C  | Soma, dend.    | 0.0003–0.0005  | L-type, HVA |
| Leak    | —        | All            | 0.0003         | Passive |

### 2.3 Cable Equation

Multi-compartment neurons are modelled via the cable equation:

```
Cm · dV/dt = (V_parent - V) / R_axial - I_ionic + I_syn + I_ext
```

Axial resistance between compartments:
```
R_axial = (4 · Ra · L) / (π · d²)
```

### 2.4 Synapse Models

Double-exponential conductance kinetics:
```
g(t) = g_max · (exp(-t/τ_decay) - exp(-t/τ_rise))
I_syn = g(t) · (V - E_rev)
```

Standard synapse parameters:

| Type   | τ_rise (ms) | τ_decay (ms) | E_rev (mV) | Notes |
|--------|-------------|--------------|------------|-------|
| AMPA   | 0.2         | 2.0          | 0          | Fast excitatory |
| NMDA   | 2.0         | 65.0         | 0          | Mg²⁺ block |
| GABA-A | 0.5         | 5.0          | −70        | Fast inhibitory |
| GABA-B | 15.0        | 150.0        | −90        | Slow inhibitory |

NMDA Mg²⁺ block (Jahr & Stevens, 1990):
```
B(V) = 1 / (1 + [Mg²⁺]/3.57 · exp(-0.062·V))
```

---

## 3. Simulation Engine

### 3.1 Time Integration

The default integrator is 4th-order Runge-Kutta (RK4):

```
k1 = f(t,    y)
k2 = f(t+h/2, y + h/2·k1)
k3 = f(t+h/2, y + h/2·k2)
k4 = f(t+h,   y + h·k3)
y(t+h) = y + (h/6)·(k1 + 2k2 + 2k3 + k4)
```

**Time step recommendation**: dt = 0.025 ms (40 kHz sampling) for Hodgkin–Huxley
models. Use dt ≤ 0.01 ms for high-frequency phenomena (fast Na channels at AIS).

**Numerical stability**: The Courant condition for cable equation:
```
Δt < (Cm · Δx²) / (2 · Ra)
```
Is automatically satisfied for typical dendrite parameters at dt = 0.025 ms.

### 3.2 Spike Event System

NeuroSim uses a **time-driven simulation with event-driven spike delivery**:

1. At each time step, all neurons are integrated in parallel
2. New spikes are detected at the soma (threshold crossing)
3. SpikeEvents are pushed to a global priority queue with `t_delivery = t_spike + delay`
4. At each step, all events with `t_delivery ≤ t_current` are popped and delivered

This gives O(log N_spikes) spike delivery cost vs O(N_neurons) for naive all-to-all.

### 3.3 Checkpoint System

Checkpoints are saved as compressed pickle files at configurable intervals:
```
checkpoints/checkpoint_step_00040000.pkl
```

Restore from checkpoint:
```bash
neurosim simulate --config config.yaml --checkpoint checkpoints/checkpoint_step_00040000.pkl
```

---

## 4. HPC & Parallelization

### 4.1 Thread-level parallelism (single node)

Python `ThreadPoolExecutor` distributes neuron integration across CPU cores.
The GIL is released during NumPy/Numba computation, enabling true parallelism.

```python
SimulationConfig(n_workers=16)  # 16 threads
```

### 4.2 GPU acceleration (CUDA)

The CUDA kernel in `cpp/kernels/hh_cuda.cu` assigns one thread per neuron:

```
GPU: 100,000 neurons × 1ms → ~5ms wall-clock (A100)
     1,000,000 neurons × 1ms → ~45ms wall-clock (8× A100)
```

Memory layout: Structure-of-Arrays (SoA) for coalesced memory access:
```cpp
float* d_V;    // [n_neurons]
float* d_m;    // [n_neurons]
float* d_h;    // [n_neurons]
float* d_n;    // [n_neurons]
```

### 4.3 MPI distributed simulation

Each MPI rank owns a partition of neurons. Synchronization occurs every
`min_delay` ms (minimum axonal delay across the entire network).

```
Rank 0: neurons [0, N/4)
Rank 1: neurons [N/4, N/2)
Rank 2: neurons [N/2, 3N/4)
Rank 3: neurons [3N/4, N)
```

Spike exchange uses non-blocking MPI_Isend/Irecv with all-to-all pattern.

**Scalability**: Linear strong scaling up to ~256 ranks (dominated by spike
communication at very high firing rates or dense connectivity).

```bash
mpirun -np 64 neurosim simulate --config config.yaml --distributed
```

### 4.4 Performance targets

| Scale                  | Hardware            | Wall time / sim-second |
|------------------------|---------------------|------------------------|
| 1 neuron (100 comps)   | 1× CPU core         | ~0.5s                  |
| 100 neurons (5 comps)  | 8× CPU cores        | ~2s                    |
| 10,000 neurons         | 1× A100 GPU         | ~8s                    |
| 100,000 neurons        | 8× A100 GPU         | ~15s                   |
| 1,000,000 neurons      | 64× A100 + 16 nodes | ~120s                  |

---

## 5. Data Architecture

### 5.1 HDF5 result format

```
results.h5
├── attrs
│   ├── t_stop        (float)  ms
│   ├── n_neurons     (int)
│   ├── n_spikes      (int)
│   └── dt            (float)
├── t                 (float32[n_t])     ms
├── V                 (float32[N,C,T])   mV  (neurons × comps × time)
└── spikes/
    ├── 0             (float64[n_spk])   ms
    ├── 1             (float64[n_spk])   ms
    └── ...
```

### 5.2 Supported input formats

| Format   | Standard          | Use case |
|----------|-------------------|----------|
| SWC      | Cannon 1900       | Single neuron morphology |
| NeuroML2 | NeuroML 2.0       | Model interchange |
| SONATA   | Allen/BBP 2020    | Large network circuits |
| HDF5     | HDF5 1.14         | Connectomics, EM data |
| NWB      | NWB 2.5           | Experimental data alignment |
| CATMAID  | JSON              | EM skeleton export |

### 5.3 Graph database (Neo4j / NetworkX)

For large-scale connectivity analysis, NeuroSim exports to Neo4j:
```cypher
(n:Neuron {id: 0, type: "L5_pyramidal"})
  -[:SYNAPSE {weight: 0.5, delay: 1.2, type: "AMPA"}]->
(m:Neuron {id: 42, type: "PV_interneuron"})
```

---

## 6. API Reference

### REST Endpoints

```
POST   /simulations           Submit simulation job
GET    /simulations/{id}      Get job status
GET    /simulations/{id}/results  Get completed results
GET    /simulations/{id}/spikes   Get spike data
DELETE /simulations/{id}      Delete job

POST   /neurons               Register neuron model
GET    /neurons/{id}          Get neuron configuration

POST   /networks              Define network topology
GET    /networks/{id}         Get network specification

GET    /health                Service health check
WS     /simulations/{id}/stream  Real-time progress
```

### Python API

```python
from neurosim.simulation.engine import (
    NeuralSimulationEngine, SimulationConfig, run_current_clamp
)
from neurosim.models.neuron import build_l5_pyramidal_cell
from neurosim.reconstruction.ingestion import ReconstructionPipeline

# Single-cell experiment
neuron  = build_l5_pyramidal_cell()
results = run_current_clamp(neuron, I_amp=2.0, t_stop_sim=1000.0)

# Network simulation
pipeline = ReconstructionPipeline()
pipeline.build_random_network(n_exc=100, n_inh=25)

cfg    = SimulationConfig(dt=0.025, t_stop=5000.0, n_workers=8)
engine = NeuralSimulationEngine(cfg)
pipeline.export_to_engine(engine)
results = engine.run()
```

---

## 7. Visualization Pipeline

### Figure types

| Figure                | Function                    | Description |
|-----------------------|-----------------------------|-------------|
| 3D Morphology         | `plot_3d_morphology()`      | SWC tree, colored by Vm or channel density |
| Spike Raster          | `plot_raster()`             | Per-neuron spike dots |
| Voltage Traces        | `plot_voltage_traces()`     | Multi-compartment Vm |
| Population PSTH       | `plot_psth()`               | Firing rate histogram ± SEM |
| Activity Heatmap      | `plot_activity_heatmap()`   | 2D: neurons × time |
| LFP + Spectrum        | `plot_lfp_and_spectrum()`   | Trace + Welch PSD |
| Connectivity Matrix   | `plot_connectivity_matrix()`| Weight matrix heatmap |

### Export

```python
import plotly.io as pio
fig = plot_raster(spike_dict, 0, 1000)
pio.write_html(fig, "raster.html")
pio.write_image(fig, "raster.png", scale=3)
```

---

## 8. ML Integration

### Parameter optimization (CMA-ES)

```python
from neurosim.ml.optimization import ParameterSpace, EvolutionaryOptimizer

space = ParameterSpace()
space.add("gNa",  50.0,  300.0)
space.add("gK",   10.0,  100.0)
space.add("gL",   0.05,  1.0,  log_scale=True)

optimizer = EvolutionaryOptimizer(space, popsize=12)
result    = optimizer.optimize(objective, n_generations=100)
```

### Dimensionality reduction

```python
from neurosim.ml.optimization import NeuralManifoldAnalysis

# activity: (n_neurons, n_timesteps)
analysis   = NeuralManifoldAnalysis(method="pca", n_components=3)
embedding  = analysis.fit_transform(activity)   # (n_t, 3)
```

---

## 9. Research Workflows

### Workflow 1: F-I curve

```bash
neurosim simulate --config configs/fi_curve.yaml --output results/fi/
neurosim analyze  --results results/fi/results.h5 --psth
```

### Workflow 2: Gamma oscillations in E-I network

```bash
neurosim simulate --config configs/ei_gamma.yaml --n-workers 8
neurosim analyze  --results results/ei_gamma/results.h5 --all
neurosim dashboard --results-dir results/
```

### Workflow 3: Large-scale HPC simulation

```bash
# Submit to SLURM
sbatch scripts/submit_hpc.sh

# Or run locally with MPI
mpirun -np 32 neurosim simulate \
    --config configs/cortical_column_10k.yaml \
    --distributed
```

### Workflow 4: Parameter calibration

```bash
neurosim optimize \
    --config configs/optimization.yaml \
    --n-generations 200 \
    --popsize 20 \
    --output calibrated_params.json
```

---

## 10. Developer Guide

### Environment setup

```bash
# Clone repository
git clone https://github.com/your-org/neurosim
cd neurosim

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install in development mode
pip install -e ".[dev,ml]"

# Install pre-commit hooks
pre-commit install

# Build C++ core (optional, for GPU/MPI features)
cmake -B build -DENABLE_CUDA=OFF -DENABLE_MPI=ON
cmake --build build -j$(nproc)
```

### Running tests

```bash
# Unit tests
pytest tests/test_neuron_models.py -v

# Integration tests
pytest tests/test_integration.py -v --timeout=120

# All tests with coverage
pytest --cov=python/neurosim --cov-report=html

# Benchmarks only
pytest tests/ --benchmark-only -v
```

### Adding a new ion channel

1. Create class in `python/neurosim/models/ion_channels.py` inheriting `IonChannel`
2. Implement `steady_state()`, `derivatives()`, `_gate_product()`, `current()`
3. Register in `CHANNEL_REGISTRY`
4. Add tests in `tests/test_neuron_models.py`
5. Add to CUDA kernel in `cpp/kernels/hh_cuda.cu` if GPU support needed

### Code style

- Black formatting: `black python/`
- Ruff linting: `ruff check python/`
- Type hints required on all public functions
- Docstrings: Google style
- Tests: pytest with class-based organisation

### Contribution workflow

1. Fork repository
2. Create feature branch: `git checkout -b feature/calcium-wave-model`
3. Implement changes with tests
4. Run full test suite: `pytest tests/ -x`
5. Open pull request with description linking to relevant neuroscience literature
