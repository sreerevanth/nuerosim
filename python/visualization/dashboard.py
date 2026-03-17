"""
neurosim/visualization/dashboard.py

Interactive visualization dashboard built with Plotly Dash.

Panels:
  1.  3D Neuron Morphology Explorer  — SWC tree with channel density coloring
  2.  Spike Raster Plot              — per-neuron spike times
  3.  Voltage Traces                 — multi-compartment Vm
  4.  Population PSTH                — firing rate histogram
  5.  Activity Heatmap               — spatial activity map
  6.  LFP / Power Spectrum           — oscillation analysis
  7.  Connectivity Graph             — circular or force-directed
  8.  Parameter Explorer             — live parameter sweeps

Run with:
    python -m neurosim.visualization.dashboard --results-dir results/
"""

from __future__ import annotations
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, List
import logging

logger = logging.getLogger("neurosim.viz")

# ─────────────────────────────────────────────────────────────
#  Plotly figure builders (usable without Dash)
# ─────────────────────────────────────────────────────────────
def plot_3d_morphology(compartments: list,
                       values: Optional[np.ndarray] = None,
                       colorscale: str = "Viridis",
                       title: str = "Neuron Morphology") -> dict:
    """
    Build a 3D Plotly scatter/line figure of neuron compartments.
    compartments: list of Compartment objects
    values:       per-compartment scalar (e.g. voltage, Ca2+ for coloring)
    Returns a Plotly figure dict.
    """
    import plotly.graph_objects as go

    if not compartments:
        return go.Figure().to_dict()

    x_pts, y_pts, z_pts, colors, texts = [], [], [], [], []
    edge_x, edge_y, edge_z = [], [], []

    comp_map = {c.id: c for c in compartments}

    for i, comp in enumerate(compartments):
        x_pts.append(comp.x)
        y_pts.append(comp.y)
        z_pts.append(comp.z)
        color_val = float(values[i]) if values is not None else comp.diameter
        colors.append(color_val)
        texts.append(
            f"ID:{comp.id} Type:{comp.type.name}<br>"
            f"d={comp.diameter:.1f}μm L={comp.length:.1f}μm"
        )

        # Draw edge to parent
        if comp.parent_id is not None and comp.parent_id in comp_map:
            par = comp_map[comp.parent_id]
            edge_x += [par.x, comp.x, None]
            edge_y += [par.y, comp.y, None]
            edge_z += [par.z, comp.z, None]

    # Color label
    color_label = "Vm (mV)" if values is not None else "Diameter (μm)"

    fig = go.Figure()

    # Dendrite/axon edges
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="rgba(150,150,150,0.4)", width=2),
        name="Branches",
        hoverinfo="skip",
    ))

    # Compartment nodes
    fig.add_trace(go.Scatter3d(
        x=x_pts, y=y_pts, z=z_pts,
        mode="markers",
        marker=dict(
            size=[max(3, c.diameter * 0.4) for c in compartments],
            color=colors,
            colorscale=colorscale,
            colorbar=dict(title=color_label, thickness=12),
            showscale=True,
            opacity=0.85,
        ),
        text=texts,
        hoverinfo="text",
        name="Compartments",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis_title="x (μm)",
            yaxis_title="y (μm)",
            zaxis_title="z (μm)",
            bgcolor="rgb(15,15,25)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        ),
        paper_bgcolor="rgb(20,20,30)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
    )
    return fig.to_dict()


def plot_raster(spike_dict: Dict[int, List[float]],
                t_start: float = 0.0,
                t_stop: float = 1000.0,
                title: str = "Spike Raster") -> dict:
    """
    Interactive spike raster plot.
    spike_dict: {neuron_id: [spike_time_ms, ...]}
    """
    import plotly.graph_objects as go

    n_neurons  = len(spike_dict)
    sorted_ids = sorted(spike_dict.keys())

    # Color neurons by excitatory (blue) / inhibitory (red) if >80 neurons
    # Otherwise use a gradient
    scatter_x, scatter_y = [], []
    for row_idx, nid in enumerate(sorted_ids):
        times = [t for t in spike_dict[nid] if t_start <= t <= t_stop]
        scatter_x.extend(times)
        scatter_y.extend([row_idx] * len(times))

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=scatter_x,
        y=scatter_y,
        mode="markers",
        marker=dict(
            size=2,
            color=scatter_y,
            colorscale="Turbo",
            opacity=0.7,
        ),
        hovertemplate="Neuron %{y}<br>t = %{x:.2f} ms<extra></extra>",
        name="Spikes",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(title="Time (ms)", range=[t_start, t_stop]),
        yaxis=dict(title="Neuron index", range=[-0.5, n_neurons - 0.5]),
        paper_bgcolor="rgb(20,20,30)",
        plot_bgcolor="rgb(15,15,25)",
        font=dict(color="white"),
        height=350,
        margin=dict(l=60, r=20, t=40, b=50),
    )
    return fig.to_dict()


def plot_voltage_traces(t: np.ndarray,
                        V: np.ndarray,
                        neuron_ids: List[int],
                        comp_idx: int = 0,
                        max_traces: int = 10,
                        title: str = "Voltage Traces") -> dict:
    """
    Plot soma voltage traces for multiple neurons.
    V: shape (n_neurons, n_comps, n_timesteps)
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    n_plot = min(max_traces, len(neuron_ids))

    colorscale = [
        f"hsl({int(360 * i / n_plot)}, 70%, 60%)"
        for i in range(n_plot)
    ]

    for i in range(n_plot):
        nid   = neuron_ids[i]
        v_trace = V[i, comp_idx, :]
        # Downsample for performance
        stride = max(1, len(t) // 4000)
        fig.add_trace(go.Scattergl(
            x=t[::stride],
            y=v_trace[::stride],
            mode="lines",
            name=f"Neuron {nid}",
            line=dict(color=colorscale[i], width=1),
            hovertemplate=f"N{nid}: %{{y:.1f}} mV @ %{{x:.2f}} ms<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(title="Time (ms)"),
        yaxis=dict(title="Vm (mV)"),
        paper_bgcolor="rgb(20,20,30)",
        plot_bgcolor="rgb(15,15,25)",
        font=dict(color="white"),
        height=300,
        showlegend=n_plot <= 5,
        margin=dict(l=60, r=20, t=40, b=50),
    )
    return fig.to_dict()


def plot_psth(bin_centers: np.ndarray,
              rate: np.ndarray,
              sem: Optional[np.ndarray] = None,
              title: str = "Population PSTH") -> dict:
    """Population peri-stimulus time histogram."""
    import plotly.graph_objects as go

    fig = go.Figure()

    if sem is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([bin_centers, bin_centers[::-1]]),
            y=np.concatenate([rate + sem, (rate - sem)[::-1]]),
            fill="toself",
            fillcolor="rgba(99, 172, 255, 0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="±SEM",
        ))

    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=rate,
        mode="lines",
        line=dict(color="#63acff", width=2),
        name="Mean rate",
        hovertemplate="%{x:.1f} ms: %{y:.2f} Hz<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(title="Time (ms)"),
        yaxis=dict(title="Firing rate (Hz)"),
        paper_bgcolor="rgb(20,20,30)",
        plot_bgcolor="rgb(15,15,25)",
        font=dict(color="white"),
        height=280,
        margin=dict(l=60, r=20, t=40, b=50),
    )
    return fig.to_dict()


def plot_activity_heatmap(activity_matrix: np.ndarray,
                          t: np.ndarray,
                          neuron_ids: List[int],
                          title: str = "Neural Activity Heatmap") -> dict:
    """
    2D heatmap: neurons × time, colored by firing rate.
    activity_matrix: shape (n_neurons, n_timebins)
    """
    import plotly.graph_objects as go

    fig = go.Figure(go.Heatmap(
        z=activity_matrix,
        x=t,
        y=[f"N{nid}" for nid in neuron_ids],
        colorscale="Inferno",
        colorbar=dict(title="Rate (Hz)", thickness=12),
        hovertemplate="Neuron %{y}<br>t=%{x:.1f}ms<br>Rate=%{z:.1f}Hz<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(title="Time (ms)"),
        yaxis=dict(title="Neuron", autorange="reversed"),
        paper_bgcolor="rgb(20,20,30)",
        font=dict(color="white"),
        height=max(300, 10 * len(neuron_ids)),
        margin=dict(l=80, r=20, t=40, b=50),
    )
    return fig.to_dict()


def plot_lfp_and_spectrum(t: np.ndarray, lfp: np.ndarray,
                          fs: float = 1000.0,
                          title: str = "LFP & Power Spectrum") -> dict:
    """
    Two-panel figure: raw LFP trace + power spectral density.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from neurosim.analysis.spike_analysis import compute_power_spectrum

    freqs, psd = compute_power_spectrum(lfp, fs=fs)
    stride = max(1, len(t) // 5000)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["LFP Trace", "Power Spectral Density"],
        column_widths=[0.6, 0.4],
    )

    fig.add_trace(go.Scattergl(
        x=t[::stride], y=lfp[::stride],
        mode="lines",
        line=dict(color="#a0d0ff", width=1),
        name="LFP",
    ), row=1, col=1)

    # PSD up to 200 Hz
    mask = freqs <= 200
    fig.add_trace(go.Scatter(
        x=freqs[mask], y=10 * np.log10(psd[mask] + 1e-30),
        mode="lines",
        line=dict(color="#ffb347", width=2),
        name="PSD",
        fill="tozeroy",
        fillcolor="rgba(255, 179, 71, 0.2)",
    ), row=1, col=2)

    # Band annotations
    bands = {"θ": (4, 12), "β": (12, 30), "γ": (30, 80)}
    for band, (lo, hi) in bands.items():
        fig.add_vrect(x0=lo, x1=hi, row=1, col=2,
                      fillcolor="rgba(255,255,255,0.05)",
                      annotation_text=band, annotation_position="top left",
                      line_width=0)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        paper_bgcolor="rgb(20,20,30)",
        plot_bgcolor="rgb(15,15,25)",
        font=dict(color="white"),
        height=300,
        margin=dict(l=60, r=20, t=60, b=50),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="mV", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="Power (dB)", row=1, col=2)

    return fig.to_dict()


def plot_connectivity_matrix(adj: np.ndarray,
                             labels: Optional[List[str]] = None,
                             title: str = "Connectivity Matrix") -> dict:
    """Synaptic weight matrix heatmap."""
    import plotly.graph_objects as go

    n = adj.shape[0]
    lbl = labels or [str(i) for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=adj,
        x=lbl, y=lbl,
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="Weight", thickness=12),
        hovertemplate="Pre:%{y} → Post:%{x}<br>w=%{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(title="Post-synaptic", tickangle=-45),
        yaxis=dict(title="Pre-synaptic", autorange="reversed"),
        paper_bgcolor="rgb(20,20,30)",
        font=dict(color="white"),
        height=max(350, 8 * n),
        margin=dict(l=80, r=20, t=40, b=80),
    )
    return fig.to_dict()


# ─────────────────────────────────────────────────────────────
#  Dash application
# ─────────────────────────────────────────────────────────────
def create_dashboard(results_dir: str = "results") -> object:
    """
    Create and return the Dash application object.
    Call app.run_server() to launch.
    """
    try:
        import dash
        from dash import dcc, html, Input, Output, State, callback
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Install dash: pip install dash>=2.11")

    results_path = Path(results_dir)

    app = dash.Dash(
        __name__,
        title="NeuroSim Dashboard",
        meta_tags=[{"name": "viewport",
                    "content": "width=device-width, initial-scale=1"}],
    )

    # ── Layout ────────────────────────────────────────────────
    app.layout = html.Div(
        style={"backgroundColor": "#0d0d1a", "minHeight": "100vh",
               "fontFamily": "Inter, system-ui, sans-serif", "color": "white"},
        children=[

            # Header
            html.Div(style={"padding": "16px 24px",
                            "borderBottom": "1px solid #2a2a3e",
                            "display": "flex", "alignItems": "center",
                            "gap": "16px"},
                     children=[
                         html.H1("NeuroSim",
                                 style={"margin": 0, "fontSize": "1.5rem",
                                        "color": "#63acff"}),
                         html.Span("Brain Simulation Platform",
                                   style={"color": "#888", "fontSize": "0.9rem"}),
                         html.Div(style={"marginLeft": "auto"},
                                  children=[
                                      dcc.Interval(id="live-update",
                                                   interval=2000,
                                                   disabled=True),
                                  ]),
                     ]),

            # Controls row
            html.Div(style={"padding": "12px 24px",
                            "borderBottom": "1px solid #1e1e2e",
                            "display": "flex", "gap": "16px",
                            "alignItems": "center"},
                     children=[
                         html.Label("Results file:",
                                    style={"color": "#aaa", "fontSize": "0.85rem"}),
                         dcc.Dropdown(
                             id="results-dropdown",
                             options=[
                                 {"label": str(p.name), "value": str(p)}
                                 for p in results_path.glob("**/*.h5")
                             ],
                             placeholder="Select results file…",
                             style={"width": "360px",
                                    "backgroundColor": "#1a1a2e",
                                    "color": "white"},
                         ),
                         html.Label("t range (ms):",
                                    style={"color": "#aaa", "fontSize": "0.85rem"}),
                         dcc.RangeSlider(
                             id="t-range-slider",
                             min=0, max=1000, step=10,
                             value=[0, 1000],
                             marks={0: "0", 500: "500", 1000: "1000ms"},
                             tooltip={"placement": "bottom",
                                      "always_visible": False},
                         ),
                         html.Button("Load",
                                     id="load-btn",
                                     style={"backgroundColor": "#2563eb",
                                            "color": "white",
                                            "border": "none",
                                            "padding": "8px 18px",
                                            "borderRadius": "6px",
                                            "cursor": "pointer"}),
                     ]),

            # Stats bar
            html.Div(id="stats-bar",
                     style={"padding": "8px 24px",
                            "backgroundColor": "#111122",
                            "display": "flex", "gap": "32px"}),

            # Main panels grid
            html.Div(style={"padding": "16px 24px",
                            "display": "grid",
                            "gridTemplateColumns": "1fr 1fr",
                            "gap": "16px"},
                     children=[
                         # Panel 1: 3D Morphology
                         html.Div(
                             className="panel",
                             style={"backgroundColor": "#14142a",
                                    "borderRadius": "8px",
                                    "padding": "12px"},
                             children=[
                                 html.H3("Neuron Morphology",
                                         style={"margin": "0 0 8px",
                                                "fontSize": "0.95rem",
                                                "color": "#aabfff"}),
                                 dcc.Graph(id="morphology-3d",
                                           style={"height": "460px"},
                                           config={"displayModeBar": True}),
                             ]
                         ),

                         # Panel 2: Raster
                         html.Div(
                             style={"backgroundColor": "#14142a",
                                    "borderRadius": "8px",
                                    "padding": "12px"},
                             children=[
                                 html.H3("Spike Raster",
                                         style={"margin": "0 0 8px",
                                                "fontSize": "0.95rem",
                                                "color": "#aabfff"}),
                                 dcc.Graph(id="raster-plot",
                                           config={"displayModeBar": False}),
                             ]
                         ),

                         # Panel 3: Voltage traces
                         html.Div(
                             style={"backgroundColor": "#14142a",
                                    "borderRadius": "8px",
                                    "padding": "12px"},
                             children=[
                                 html.H3("Voltage Traces",
                                         style={"margin": "0 0 8px",
                                                "fontSize": "0.95rem",
                                                "color": "#aabfff"}),
                                 dcc.Graph(id="voltage-traces",
                                           config={"displayModeBar": False}),
                             ]
                         ),

                         # Panel 4: PSTH
                         html.Div(
                             style={"backgroundColor": "#14142a",
                                    "borderRadius": "8px",
                                    "padding": "12px"},
                             children=[
                                 html.H3("Population PSTH",
                                         style={"margin": "0 0 8px",
                                                "fontSize": "0.95rem",
                                                "color": "#aabfff"}),
                                 dcc.Graph(id="psth-plot",
                                           config={"displayModeBar": False}),
                             ]
                         ),

                         # Panel 5: Activity Heatmap (full width)
                         html.Div(
                             style={"backgroundColor": "#14142a",
                                    "borderRadius": "8px",
                                    "padding": "12px",
                                    "gridColumn": "1 / -1"},
                             children=[
                                 html.H3("Activity Heatmap",
                                         style={"margin": "0 0 8px",
                                                "fontSize": "0.95rem",
                                                "color": "#aabfff"}),
                                 dcc.Graph(id="heatmap-plot",
                                           config={"displayModeBar": False}),
                             ]
                         ),
                     ]),

            # Hidden data store
            dcc.Store(id="sim-data-store"),
        ]
    )

    # ── Callbacks ─────────────────────────────────────────────
    @app.callback(
        Output("sim-data-store",   "data"),
        Output("stats-bar",        "children"),
        Input("load-btn",          "n_clicks"),
        State("results-dropdown",  "value"),
        State("t-range-slider",    "value"),
        prevent_initial_call=True,
    )
    def load_results(n_clicks, results_file, t_range):
        if not results_file:
            return {}, []

        try:
            import h5py
            t_start, t_stop = t_range

            with h5py.File(results_file, "r") as f:
                spikes = {}
                if "spikes" in f:
                    for nid_str in f["spikes"]:
                        spikes[int(nid_str)] = f["spikes"][nid_str][:].tolist()

                V, t_arr = None, None
                if "V" in f:
                    V_h5  = f["V"][:]
                    t_arr = f["t"][:].tolist()
                    V     = V_h5.tolist()

                attrs = dict(f.attrs)

            total_spikes = sum(len(v) for v in spikes.values())
            n_neurons    = len(spikes)
            duration     = t_stop - t_start
            mean_rate    = total_spikes / max(1, n_neurons) / (duration / 1000.0)

            stats = [
                _stat_chip(f"{n_neurons}", "Neurons"),
                _stat_chip(f"{total_spikes:,}", "Total Spikes"),
                _stat_chip(f"{mean_rate:.2f} Hz", "Mean Rate"),
                _stat_chip(f"{duration:.0f} ms", "Duration"),
            ]

            data = {
                "spikes":   {str(k): v for k, v in spikes.items()},
                "V":        V,
                "t":        t_arr,
                "t_start":  t_start,
                "t_stop":   t_stop,
            }
            return data, stats

        except Exception as e:
            logger.exception(f"Failed to load {results_file}: {e}")
            return {}, [html.Span(f"Error: {e}", style={"color": "#f87171"})]

    @app.callback(
        Output("raster-plot",    "figure"),
        Output("psth-plot",      "figure"),
        Output("heatmap-plot",   "figure"),
        Output("voltage-traces", "figure"),
        Input("sim-data-store",  "data"),
        prevent_initial_call=True,
    )
    def update_plots(data):
        import plotly.graph_objects as go

        empty = go.Figure(layout=dict(
            paper_bgcolor="rgb(20,20,30)",
            plot_bgcolor="rgb(15,15,25)",
            font=dict(color="#555"),
        )).to_dict()

        if not data:
            return empty, empty, empty, empty

        spikes  = {int(k): v for k, v in data.get("spikes", {}).items()}
        t_start = data.get("t_start", 0.0)
        t_stop  = data.get("t_stop",  1000.0)

        # Raster
        raster  = plot_raster(spikes, t_start, t_stop)

        # PSTH
        from neurosim.analysis.spike_analysis import compute_psth
        spike_trains = list(spikes.values())
        if spike_trains:
            psth_data = compute_psth(spike_trains, t_start, t_stop, bin_size=20.0)
            psth_fig  = plot_psth(psth_data.bin_centers,
                                   psth_data.rate, psth_data.sem)
        else:
            psth_fig = empty

        # Heatmap (binned activity)
        bin_size   = max(10.0, (t_stop - t_start) / 200)
        bins       = np.arange(t_start, t_stop + bin_size, bin_size)
        nids       = sorted(spikes.keys())[:50]  # max 50 for heatmap
        act_matrix = np.zeros((len(nids), len(bins) - 1))
        for row, nid in enumerate(nids):
            cnt, _ = np.histogram(spikes[nid], bins=bins)
            act_matrix[row] = cnt / (bin_size / 1000.0)
        heatmap = plot_activity_heatmap(
            act_matrix, 0.5 * (bins[:-1] + bins[1:]), nids
        )

        # Voltage
        V_data = data.get("V")
        t_data = data.get("t")
        if V_data and t_data:
            V   = np.array(V_data)
            t   = np.array(t_data)
            mask = (t >= t_start) & (t <= t_stop)
            volt_fig = plot_voltage_traces(
                t[mask], V[:, :, mask],
                neuron_ids=list(range(min(8, V.shape[0])))
            )
        else:
            volt_fig = empty

        return raster, psth_fig, heatmap, volt_fig

    return app


def _stat_chip(value: str, label: str) -> object:
    """Helper to create a stats chip for the stats bar."""
    try:
        from dash import html
    except ImportError:
        return {}
    return html.Div(style={"textAlign": "center"}, children=[
        html.Div(value, style={"fontSize": "1.1rem", "fontWeight": "600",
                               "color": "#63acff"}),
        html.Div(label, style={"fontSize": "0.75rem", "color": "#666"}),
    ])


# ─────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="NeuroSim Visualization Dashboard")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing HDF5 result files")
    parser.add_argument("--port",  type=int, default=8050)
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    dash_app = create_dashboard(results_dir=args.results_dir)
    print(f"\n  NeuroSim Dashboard running at http://localhost:{args.port}\n")
    dash_app.run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
