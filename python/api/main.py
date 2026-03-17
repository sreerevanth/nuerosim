"""
neurosim/api/main.py

FastAPI-based REST API for the NeuroSim platform.

Endpoints:
  POST /simulations           — Submit simulation job
  GET  /simulations/{id}      — Get job status and results
  POST /neurons               — Create/register neuron model
  GET  /neurons/{id}          — Get neuron configuration
  POST /networks              — Submit network specification
  GET  /results/{id}/voltage  — Download voltage traces
  GET  /results/{id}/spikes   — Download spike data
  GET  /results/{id}/analysis — Get pre-computed analysis
  WS   /simulations/{id}/stream — Real-time simulation streaming
"""

from __future__ import annotations
import uuid
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

logger = logging.getLogger("neurosim.api")

app = FastAPI(
    title="NeuroSim Platform API",
    description="Large-scale neural simulation REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
#  In-memory job store (replace with Redis in production)
# ─────────────────────────────────────────────────────────────
class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class SimJob:
    def __init__(self, job_id: str, config: dict):
        self.job_id     = job_id
        self.config     = config
        self.status     = JobStatus.PENDING
        self.progress   = 0.0
        self.results    = None
        self.error      = None
        self.created_at = time.time()
        self.started_at = None
        self.ended_at   = None
        self._ws_clients: List[WebSocket] = []


_jobs: Dict[str, SimJob] = {}
_neurons: Dict[str, dict] = {}
_networks: Dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────
#  Request / Response models
# ─────────────────────────────────────────────────────────────
class ChannelSpec(BaseModel):
    name:   str
    g_max:  float = 1.0
    E_rev:  Optional[float] = None


class CompartmentSpec(BaseModel):
    id:       int
    type:     str = "soma"
    length:   float = 20.0
    diameter: float = 20.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    channels: List[ChannelSpec] = Field(default_factory=list)
    parent_id: Optional[int] = None


class NeuronSpec(BaseModel):
    neuron_type:  str = "custom"          # "l5_pyramidal" | "pv_interneuron" | "custom"
    compartments: List[CompartmentSpec] = Field(default_factory=list)
    temperature:  float = 37.0
    V_init:       float = -65.0


class SynapseSpec(BaseModel):
    src_neuron:  int
    tgt_neuron:  int
    syn_type:    str = "ampa"
    weight:      float = 0.5
    delay:       float = 1.0
    tgt_comp:    int = 0


class NetworkSpec(BaseModel):
    name:        str
    n_excitatory: int = 100
    n_inhibitory: int = 25
    connectivity_type: str = "random"   # "random" | "small_world" | "custom"
    p_ee: float = 0.1
    p_ei: float = 0.5
    p_ie: float = 0.5
    p_ii: float = 0.1
    custom_synapses: List[SynapseSpec] = Field(default_factory=list)


class SimulationRequest(BaseModel):
    network_id:  Optional[str] = None
    neuron_ids:  List[str] = Field(default_factory=list)
    t_stop:      float = 1000.0         # ms
    dt:          float = 0.025          # ms
    integrator:  str = "rk4"
    temperature: float = 37.0
    current_clamps: List[dict] = Field(default_factory=list)
    distributed: bool = False
    n_workers:   int = 4
    record_dt:   float = 0.1

    @validator("dt")
    def dt_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("dt must be positive")
        return v

    @validator("t_stop")
    def t_stop_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("t_stop must be positive")
        return v


class SimulationResponse(BaseModel):
    job_id:    str
    status:    JobStatus
    message:   str


class JobStatusResponse(BaseModel):
    job_id:    str
    status:    JobStatus
    progress:  float
    created_at: float
    started_at: Optional[float]
    ended_at:   Optional[float]
    n_neurons:  Optional[int]
    n_spikes:   Optional[int]
    mean_rate_hz: Optional[float]
    error:      Optional[str]


# ─────────────────────────────────────────────────────────────
#  Background simulation runner
# ─────────────────────────────────────────────────────────────
async def run_simulation_task(job: SimJob) -> None:
    """Run simulation in background task."""
    job.status     = JobStatus.RUNNING
    job.started_at = time.time()

    try:
        from neurosim.simulation.engine import (
            NeuralSimulationEngine, SimulationConfig
        )
        from neurosim.reconstruction.ingestion import ReconstructionPipeline

        cfg = SimulationConfig(
            dt          = job.config.get("dt", 0.025),
            t_stop      = job.config.get("t_stop", 1000.0),
            integrator  = job.config.get("integrator", "rk4"),
            record_dt   = job.config.get("record_dt", 0.1),
            n_workers   = job.config.get("n_workers", 4),
            temperature = job.config.get("temperature", 37.0),
        )
        engine = NeuralSimulationEngine(cfg)

        # Build network
        pipeline = ReconstructionPipeline()
        network_id = job.config.get("network_id")
        if network_id and network_id in _networks:
            net_spec = _networks[network_id]
            pipeline.build_random_network(
                n_exc = net_spec.get("n_excitatory", 100),
                n_inh = net_spec.get("n_inhibitory", 25),
                p_ee  = net_spec.get("p_ee", 0.1),
                p_ei  = net_spec.get("p_ei", 0.5),
                p_ie  = net_spec.get("p_ie", 0.5),
                p_ii  = net_spec.get("p_ii", 0.1),
            )
        else:
            pipeline.build_random_network(n_exc=50, n_inh=12)

        pipeline.export_to_engine(engine)

        # Apply current clamps
        for clamp in job.config.get("current_clamps", []):
            nid   = clamp.get("neuron_id", 0)
            I_amp = clamp.get("I_amp", 2.0)
            t0    = clamp.get("t_start", 100.0)
            t1    = clamp.get("t_stop", 500.0)
            engine.clamp_current(nid, lambda t: I_amp if t0 <= t <= t1 else 0.0)

        # Progress callback
        def progress_cb(pct: float, t: float):
            job.progress = pct
            asyncio.get_event_loop().call_soon_threadsafe(
                lambda: _broadcast_progress(job, pct, t)
            )

        # Run
        results = engine.run(progress_callback=progress_cb)

        job.results  = {
            "n_neurons":      results.get("n_neurons"),
            "n_connections":  results.get("n_connections"),
            "total_spikes":   results.get("total_spikes"),
            "mean_rate_hz":   results.get("mean_firing_rate_hz"),
            "wall_time_s":    results.get("wall_time_s"),
            "t_stop_ms":      cfg.t_stop,
        }
        # Store full recording in job for download
        job._full_results = results
        job.status    = JobStatus.COMPLETED
        job.progress  = 100.0

    except Exception as e:
        logger.exception(f"Simulation job {job.job_id} failed: {e}")
        job.status = JobStatus.FAILED
        job.error  = str(e)

    finally:
        job.ended_at = time.time()


def _broadcast_progress(job: SimJob, pct: float, t: float) -> None:
    """Non-blocking broadcast to WebSocket clients."""
    pass  # implemented in WebSocket endpoint


# ─────────────────────────────────────────────────────────────
#  API endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0", "n_jobs": len(_jobs)}


@app.post("/neurons", response_model=dict)
async def create_neuron(spec: NeuronSpec):
    """Register a neuron model specification."""
    nid = str(uuid.uuid4())
    _neurons[nid] = spec.dict()
    return {"neuron_id": nid, "type": spec.neuron_type}


@app.get("/neurons/{neuron_id}")
async def get_neuron(neuron_id: str):
    if neuron_id not in _neurons:
        raise HTTPException(status_code=404, detail="Neuron not found")
    return _neurons[neuron_id]


@app.post("/networks", response_model=dict)
async def create_network(spec: NetworkSpec):
    """Define a network topology."""
    net_id = str(uuid.uuid4())
    _networks[net_id] = spec.dict()
    return {"network_id": net_id, "name": spec.name}


@app.get("/networks/{network_id}")
async def get_network(network_id: str):
    if network_id not in _networks:
        raise HTTPException(status_code=404, detail="Network not found")
    return _networks[network_id]


@app.post("/simulations", response_model=SimulationResponse)
async def submit_simulation(req: SimulationRequest,
                             background_tasks: BackgroundTasks):
    """Submit a simulation job."""
    job_id = str(uuid.uuid4())
    job    = SimJob(job_id, req.dict())
    _jobs[job_id] = job

    background_tasks.add_task(run_simulation_task, job)

    return SimulationResponse(
        job_id  = job_id,
        status  = JobStatus.PENDING,
        message = "Simulation job submitted"
    )


@app.get("/simulations/{job_id}", response_model=JobStatusResponse)
async def get_simulation_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _jobs[job_id]
    r   = job.results or {}
    return JobStatusResponse(
        job_id      = job.job_id,
        status      = job.status,
        progress    = job.progress,
        created_at  = job.created_at,
        started_at  = job.started_at,
        ended_at    = job.ended_at,
        n_neurons   = r.get("n_neurons"),
        n_spikes    = r.get("total_spikes"),
        mean_rate_hz = r.get("mean_rate_hz"),
        error       = job.error,
    )


@app.get("/simulations/{job_id}/results")
async def get_simulation_results(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _jobs[job_id]
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400,
                            detail=f"Job not completed (status: {job.status})")
    return job.results


@app.get("/simulations/{job_id}/spikes")
async def get_spike_data(job_id: str):
    """Return spike times for all neurons."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _jobs[job_id]
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")
    full = getattr(job, "_full_results", {})
    spikes = full.get("spikes", {})
    return {"job_id": job_id, "spikes": {str(k): v for k, v in spikes.items()}}


@app.delete("/simulations/{job_id}")
async def delete_simulation(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    del _jobs[job_id]
    return {"deleted": job_id}


@app.get("/simulations")
async def list_simulations(limit: int = 20, status: Optional[str] = None):
    """List recent simulation jobs."""
    jobs = list(_jobs.values())
    if status:
        jobs = [j for j in jobs if j.status == status]
    jobs = sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]
    return [{"job_id": j.job_id, "status": j.status,
             "progress": j.progress, "created_at": j.created_at}
            for j in jobs]


# WebSocket for real-time simulation streaming
@app.websocket("/simulations/{job_id}/stream")
async def simulation_stream(websocket: WebSocket, job_id: str):
    """Stream simulation progress updates via WebSocket."""
    await websocket.accept()
    if job_id not in _jobs:
        await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return

    job = _jobs[job_id]
    job._ws_clients.append(websocket)

    try:
        while job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            await websocket.send_json({
                "job_id":   job_id,
                "status":   job.status,
                "progress": job.progress,
            })
            await asyncio.sleep(0.5)

        await websocket.send_json({
            "job_id":   job_id,
            "status":   job.status,
            "progress": 100.0,
            "results":  job.results,
        })
    except Exception:
        pass
    finally:
        job._ws_clients.remove(websocket)
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
