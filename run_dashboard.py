"""
run_dashboard.py  —  NeuroSim self-contained Windows launcher.
All simulation code is embedded here. No imports from the neurosim package needed.
"""

import sys, os, time, webbrowser, threading, json, math
from pathlib import Path

# ── Install check ─────────────────────────────────────────────────────────
def check_and_install():
    import subprocess
    needed = ["numpy", "scipy", "plotly", "dash", "rich"]
    for pkg in needed:
        try:
            __import__(pkg)
        except ImportError:
            print(f"  Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

check_and_install()

import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    def log(msg): console.print(msg)
except Exception:
    def log(msg):
        import re
        print(re.sub(r'\[.*?\]', '', str(msg)))

# ═════════════════════════════════════════════════════════════════════════════
#  EMBEDDED SIMULATION CODE  (self-contained, no package imports needed)
# ═════════════════════════════════════════════════════════════════════════════

Q10 = 3.0
DEFAULT_T = 37.0

def q10_scale(rate, T, T_ref=6.3):
    return rate * Q10 ** ((T - T_ref) / 10.0)

# ── Na channel rates ──────────────────────────────────────────────────────
def alpha_m(V):
    dV = V + 40.0
    return q10_scale(1.0 if abs(dV) < 1e-7 else 0.1*dV/(1-np.exp(-dV/10)), DEFAULT_T)

def beta_m(V):
    return q10_scale(4.0 * np.exp(-(V+65)/18), DEFAULT_T)

def alpha_h(V):
    return q10_scale(0.07 * np.exp(-(V+65)/20), DEFAULT_T)

def beta_h(V):
    return q10_scale(1.0 / (1 + np.exp(-(V+35)/10)), DEFAULT_T)

def alpha_n(V):
    dV = V + 55.0
    return q10_scale(0.1 if abs(dV) < 1e-7 else 0.01*dV/(1-np.exp(-dV/10)), DEFAULT_T)

def beta_n(V):
    return q10_scale(0.125 * np.exp(-(V+65)/80), DEFAULT_T)

# ── Single-compartment HH neuron state ───────────────────────────────────
def hh_steady_state(V=-65.0):
    am, bm = alpha_m(V), beta_m(V)
    ah, bh = alpha_h(V), beta_h(V)
    an, bn = alpha_n(V), beta_n(V)
    return {
        "V": V,
        "m": am/(am+bm),
        "h": ah/(ah+bh),
        "n": an/(an+bn),
    }

def hh_derivatives(state, I_ext, gNa=120.0, gK=36.0, gL=0.3,
                   ENa=50.0, EK=-77.0, EL=-54.387, Cm=1.0):
    V, m, h, n = state["V"], state["m"], state["h"], state["n"]
    INa = gNa * m**3 * h * (V - ENa)
    IK  = gK  * n**4     * (V - EK)
    IL  = gL              * (V - EL)
    return {
        "V": (I_ext - INa - IK - IL) / Cm,
        "m": alpha_m(V)*(1-m) - beta_m(V)*m,
        "h": alpha_h(V)*(1-h) - beta_h(V)*h,
        "n": alpha_n(V)*(1-n) - beta_n(V)*n,
    }

def rk4_step(state, I_ext, dt):
    def add(s, ds, h):
        return {k: s[k] + h*ds[k] for k in s}
    k1 = hh_derivatives(state,              I_ext)
    k2 = hh_derivatives(add(state, k1, dt/2), I_ext)
    k3 = hh_derivatives(add(state, k2, dt/2), I_ext)
    k4 = hh_derivatives(add(state, k3, dt),   I_ext)
    return {k: state[k] + (dt/6)*(k1[k]+2*k2[k]+2*k3[k]+k4[k]) for k in state}

def simulate_neuron(I_amp=3.0, t_start_inj=100.0, t_stop_inj=700.0,
                    t_stop=800.0, dt=0.025):
    """Simulate single HH neuron, return time array and voltage."""
    steps  = int(t_stop / dt)
    t_arr  = np.arange(steps) * dt
    V_arr  = np.zeros(steps)
    state  = hh_steady_state(-65.0)
    spikes = []
    above  = False

    for i in range(steps):
        t      = t_arr[i]
        I_ext  = I_amp if t_start_inj <= t <= t_stop_inj else 0.0
        state  = rk4_step(state, I_ext, dt)
        V_arr[i] = state["V"]
        if state["V"] > -20 and not above:
            spikes.append(t)
            above = True
        elif state["V"] < -20:
            above = False

    return t_arr, V_arr, spikes

def simulate_fi_curve():
    """F-I curve: firing rate vs injected current."""
    currents = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    rates    = []
    for I in currents:
        _, _, spikes = simulate_neuron(I_amp=I, t_start_inj=50,
                                        t_stop_inj=450, t_stop=500)
        rate = len(spikes) / 0.4   # Hz over 400ms window
        rates.append(rate)
        log(f"    I = {I:.1f} nA  ->  {rate:.1f} Hz")
    return currents, rates

def simulate_network(n_exc=48, n_inh=12, t_stop=500.0, dt=0.025, seed=42):
    """
    Simulate a simple E-I network of single-compartment HH neurons.
    Returns spike dict {neuron_id: [spike_times]}.
    """
    rng     = np.random.default_rng(seed)
    n_total = n_exc + n_inh
    steps   = int(t_stop / dt)

    # Initial states
    states = [hh_steady_state(-65.0 + rng.uniform(-2, 2)) for _ in range(n_total)]
    spikes = {i: [] for i in range(n_total)}
    above  = [False] * n_total

    # Random connectivity (sparse)
    p_connect = 0.12
    w_exc, w_inh = 0.8, 1.5
    conns = []   # (src, tgt, weight, delay_steps)
    for src in range(n_total):
        for tgt in range(n_total):
            if src == tgt: continue
            if rng.random() > p_connect: continue
            is_exc = src < n_exc
            w      = w_exc if is_exc else -w_inh
            delay  = max(1, int(rng.uniform(1, 3) / dt))
            conns.append((src, tgt, w, delay))

    # Spike delivery buffer
    I_syn    = np.zeros(n_total)
    # circular buffer for delayed spikes: [step_index] -> list of (tgt, weight)
    buf_size = 200
    spike_buf = [[] for _ in range(buf_size)]

    driven = list(range(6))   # first 6 excitatory neurons get external drive

    for step in range(steps):
        t = step * dt

        # Deliver buffered spikes
        slot = step % buf_size
        for tgt, w in spike_buf[slot]:
            I_syn[tgt] += w * 0.3
        spike_buf[slot] = []

        # Decay synaptic current
        I_syn *= np.exp(-dt / 5.0)

        # Integrate each neuron
        new_spikes = []
        for i in range(n_total):
            I_ext = 3.0 if (i in driven and 50 <= t <= 450) else 0.0
            I_ext += I_syn[i]
            # Small noise
            I_ext += rng.normal(0, 0.1)

            states[i] = rk4_step(states[i], I_ext, dt)
            V = states[i]["V"]

            if V > -20 and not above[i]:
                spikes[i].append(t)
                above[i]  = True
                new_spikes.append(i)
            elif V < -20:
                above[i]  = False

        # Queue outgoing spikes
        for src in new_spikes:
            for (s, tgt, w, delay) in conns:
                if s == src:
                    future_slot = (step + delay) % buf_size
                    spike_buf[future_slot].append((tgt, w))

    total = sum(len(v) for v in spikes.values())
    mean_rate = total / n_total / (t_stop / 1000.0)
    return spikes, n_total, len(conns), total, mean_rate


# ═════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def build_dashboard(t_arr, V_arr, spikes_single,
                    fi_currents, fi_rates,
                    net_spikes, n_neurons, n_conns, total_spikes, mean_rate):

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import dash
    from dash import dcc, html

    app = dash.Dash(__name__, title="NeuroSim Dashboard")

    # ── helpers ──────────────────────────────────────────────────────────
    def dark(title, xlab, ylab, height=320):
        return dict(
            title=dict(text=title, font=dict(size=13, color="#ccd6f6")),
            xaxis=dict(title=xlab, gridcolor="rgba(255,255,255,0.06)", color="#777"),
            yaxis=dict(title=ylab, gridcolor="rgba(255,255,255,0.06)", color="#777"),
            paper_bgcolor="#0d0d1a", plot_bgcolor="#0a0a18",
            font=dict(color="white"), height=height,
            margin=dict(l=55, r=15, t=42, b=48),
            hovermode="x unified",
        )

    def panel(title, fig, full=False):
        s = {"backgroundColor":"#14142a","borderRadius":"10px",
             "padding":"14px","border":"1px solid #1e1e3a"}
        if full: s["gridColumn"] = "1 / -1"
        return html.Div(style=s, children=[
            html.H3(title, style={"margin":"0 0 8px","fontSize":"0.85rem",
                                   "fontWeight":"600","color":"#aabfff",
                                   "textTransform":"uppercase","letterSpacing":"0.05em"}),
            dcc.Graph(figure=fig, config={"displayModeBar":True,"displaylogo":False}),
        ])

    # ── Figure 1: Voltage trace ───────────────────────────────────────────
    stride = max(1, len(t_arr)//8000)
    fig_v  = go.Figure()
    fig_v.add_vrect(x0=100, x1=700, fillcolor="rgba(99,172,255,0.06)", line_width=0,
                    annotation_text="Current on", annotation_font_color="#63acff",
                    annotation_position="top left")
    fig_v.add_trace(go.Scattergl(x=t_arr[::stride], y=V_arr[::stride],
                                  mode="lines", line=dict(color="#63acff", width=1.2),
                                  name="Vm"))
    fig_v.update_layout(**dark("Somatic Voltage — L5 Pyramidal", "Time (ms)", "Vm (mV)"))

    # ── Figure 2: Single AP zoom ──────────────────────────────────────────
    fig_ap = go.Figure()
    if spikes_single:
        t0   = spikes_single[0]
        mask = (t_arr >= t0-4) & (t_arr <= t0+14)
        fig_ap.add_trace(go.Scatter(x=t_arr[mask], y=V_arr[mask], mode="lines",
                                     line=dict(color="#f9a825", width=2.5),
                                     fill="tozeroy", fillcolor="rgba(249,168,37,0.07)"))
        fig_ap.add_hline(y=0,   line_dash="dot", line_color="rgba(255,255,255,0.2)")
        fig_ap.add_hline(y=-65, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                          annotation_text="Rest (-65 mV)", annotation_font_color="#888")
    fig_ap.update_layout(**dark("Action Potential Waveform", "Time (ms)", "Vm (mV)"))

    # ── Figure 3: F-I curve ───────────────────────────────────────────────
    fig_fi = go.Figure()
    fig_fi.add_trace(go.Scatter(
        x=fi_currents, y=fi_rates, mode="lines+markers",
        line=dict(color="#69db7c", width=2.5),
        marker=dict(size=9, color="#69db7c", line=dict(color="white", width=1.5)),
        hovertemplate="I=%{x:.1f} nA<br>Rate=%{y:.1f} Hz<extra></extra>",
    ))
    fig_fi.update_layout(**dark("F-I Curve", "Injected Current (nA)", "Firing Rate (Hz)"))

    # ── Figure 4: Network raster ──────────────────────────────────────────
    n_exc  = int(n_neurons * 0.8)
    sx, sy, sc = [], [], []
    for nid, times in net_spikes.items():
        for t in times:
            sx.append(t); sy.append(nid)
            sc.append("#63acff" if nid < n_exc else "#f87171")
    fig_r = go.Figure()
    fig_r.add_trace(go.Scattergl(x=sx, y=sy, mode="markers",
                                  marker=dict(size=2.5, color=sc, opacity=0.85),
                                  hovertemplate="N%{y} @ %{x:.1f}ms<extra></extra>"))
    fig_r.add_annotation(x=15, y=n_exc*0.05,   text="Excitatory",
                          font=dict(color="#63acff",size=11), showarrow=False)
    fig_r.add_annotation(x=15, y=n_exc+n_neurons*0.05, text="Inhibitory",
                          font=dict(color="#f87171",size=11), showarrow=False)
    fig_r.update_layout(**dark(f"Network Spike Raster  ({n_neurons} neurons)",
                                "Time (ms)", "Neuron", height=360))

    # ── Figure 5: PSTH ────────────────────────────────────────────────────
    bins     = np.arange(0, 501, 15)
    all_st   = list(net_spikes.values())
    pop_rate = np.zeros(len(bins)-1)
    for st in all_st:
        cnt, _ = np.histogram(st, bins=bins)
        pop_rate += cnt / (15/1000.0)
    pop_rate /= max(1, n_neurons)
    centers   = 0.5*(bins[:-1]+bins[1:])
    smoothed  = gaussian_filter1d(pop_rate, sigma=1.5)
    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(x=centers, y=pop_rate,
                            marker_color="rgba(99,172,255,0.2)", name="Raw"))
    fig_p.add_trace(go.Scatter(x=centers, y=smoothed, mode="lines",
                                line=dict(color="#63acff", width=2.5), name="Smoothed"))
    fig_p.update_layout(**dark("Population Firing Rate (PSTH)",
                                "Time (ms)", "Mean Rate (Hz)", height=300))

    # ── Figure 6: Heatmap ─────────────────────────────────────────────────
    nids = sorted(net_spikes.keys())[:50]
    bins2 = np.arange(0, 501, 20)
    mat  = np.zeros((len(nids), len(bins2)-1))
    for r, nid in enumerate(nids):
        cnt, _ = np.histogram(net_spikes[nid], bins=bins2)
        mat[r] = cnt / (20/1000.0)
    fig_h = go.Figure(go.Heatmap(
        z=mat, x=0.5*(bins2[:-1]+bins2[1:]),
        y=[f"N{n}" for n in nids],
        colorscale="Inferno",
        colorbar=dict(title="Hz", thickness=10,
                      tickfont=dict(color="white"),
                      titlefont=dict(color="white")),
        hovertemplate="N%{y}<br>t=%{x:.0f}ms<br>%{z:.1f}Hz<extra></extra>",
    ))
    fig_h.update_layout(**dark("Activity Heatmap", "Time (ms)", "Neuron",
                                height=max(300, 7*len(nids))))
    fig_h.update_yaxes(autorange="reversed")

    # ── Stats table ───────────────────────────────────────────────────────
    rows = [
        ("Neuron model",      "Hodgkin-Huxley (4 state variables)"),
        ("Channels",          "NaV  Kdr  Leak"),
        ("Integrator",        "4th-order Runge-Kutta"),
        ("Time step",         "0.025 ms  (40 kHz)"),
        ("Single cell spikes", str(len(spikes_single))),
        ("Single cell rate",  f"{len(spikes_single)/0.6:.1f} Hz"),
        ("AP peak voltage",   f"{V_arr.max():.1f} mV"),
        ("Network neurons",   str(n_neurons)),
        ("Network synapses",  str(n_conns)),
        ("Network spikes",    f"{total_spikes:,}"),
        ("Network mean rate", f"{mean_rate:.1f} Hz"),
    ]
    fig_s = go.Figure(go.Table(
        header=dict(values=["Parameter","Value"],
                    fill_color="#1e1e3a",
                    font=dict(color="#aabfff", size=12),
                    align="left", height=30),
        cells=dict(values=[[r[0] for r in rows],[r[1] for r in rows]],
                   fill_color=["#14142a","#0d0d1a"],
                   font=dict(color=["#888","white"], size=12),
                   align="left", height=26),
    ))
    fig_s.update_layout(paper_bgcolor="#0d0d1a",
                         margin=dict(l=0,r=0,t=4,b=0), height=350)

    # ── Stat chips ───────────────────────────────────────────────────────
    def chip(val, label):
        return html.Div(style={"textAlign":"center"}, children=[
            html.Div(val,   style={"fontSize":"1.1rem","fontWeight":700,"color":"#63acff"}),
            html.Div(label, style={"fontSize":"0.72rem","color":"#555"}),
        ])

    # ── Layout ────────────────────────────────────────────────────────────
    app.layout = html.Div(
        style={"backgroundColor":"#080818","minHeight":"100vh",
               "fontFamily":"'Segoe UI',sans-serif","color":"white","paddingBottom":"40px"},
        children=[
            # Header
            html.Div(style={
                "background":"linear-gradient(90deg,#0d0d2e,#1a1a4e)",
                "borderBottom":"1px solid #2a2a5e",
                "padding":"14px 28px","display":"flex","alignItems":"center","gap":"14px",
            }, children=[
                html.Span("🧠", style={"fontSize":"1.6rem"}),
                html.H1("NeuroSim Dashboard",
                         style={"margin":0,"fontSize":"1.3rem",
                                "color":"#63acff","fontWeight":700}),
                html.Span("Brain Simulation Platform",
                           style={"color":"#444","fontSize":"0.82rem"}),
                html.Div(style={"marginLeft":"auto","display":"flex","gap":"28px"},
                         children=[
                             chip(str(n_neurons),         "Neurons"),
                             chip(str(n_conns),           "Synapses"),
                             chip(f"{total_spikes:,}",    "Spikes"),
                             chip(f"{mean_rate:.1f} Hz",  "Mean Rate"),
                         ]),
            ]),
            # Grid
            html.Div(style={
                "padding":"18px 28px","display":"grid",
                "gridTemplateColumns":"1fr 1fr","gap":"14px",
            }, children=[
                panel("Somatic Voltage Trace",          fig_v),
                panel("Action Potential Waveform",      fig_ap),
                panel("F-I Curve",                      fig_fi),
                panel("Population Firing Rate (PSTH)",  fig_p),
                panel("Network Spike Raster",           fig_r,  full=True),
                panel("Activity Heatmap",               fig_h,  full=True),
                panel("Simulation Summary",             fig_s),
            ]),
            html.Div("NeuroSim v1.0  |  Hodgkin-Huxley 1952  |  RK4 integration  |  Plotly Dash",
                     style={"textAlign":"center","color":"#2a2a4a",
                            "fontSize":"0.76rem","marginTop":"8px"}),
        ]
    )
    return app


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

PORT = 8050

def open_browser():
    time.sleep(3)
    webbrowser.open(f"http://127.0.0.1:{PORT}")

if __name__ == "__main__":
    log("\n[bold cyan]  NeuroSim - Brain Simulation Platform[/bold cyan]")
    log("  ================================================\n")

    # ── Sim 1: Single neuron ─────────────────────────────────────────────
    log("[bold]Step 1/3:[/bold] Single neuron current clamp...")
    t0 = time.perf_counter()
    t_arr, V_arr, spikes_single = simulate_neuron(
        I_amp=3.0, t_start_inj=100, t_stop_inj=700, t_stop=800, dt=0.025
    )
    log(f"  OK  {len(spikes_single)} spikes  |  "
        f"peak {V_arr.max():.1f} mV  |  {time.perf_counter()-t0:.1f}s")

    # ── Sim 2: F-I curve ─────────────────────────────────────────────────
    log("\n[bold]Step 2/3:[/bold] F-I curve (7 current levels)...")
    t0 = time.perf_counter()
    fi_currents, fi_rates = simulate_fi_curve()
    log(f"  OK  {time.perf_counter()-t0:.1f}s")

    # ── Sim 3: Network ───────────────────────────────────────────────────
    log("\n[bold]Step 3/3:[/bold] E-I network (60 neurons, 500ms)...")
    log("  [dim](this takes ~30s — the network has 60 neurons each running HH)[/dim]")
    t0 = time.perf_counter()
    net_spikes, n_neurons, n_conns, total_spikes, mean_rate = simulate_network(
        n_exc=48, n_inh=12, t_stop=500.0, dt=0.025, seed=42
    )
    log(f"  OK  {n_neurons} neurons  |  {n_conns} synapses  |  "
        f"{total_spikes:,} spikes  |  {mean_rate:.1f} Hz  |  "
        f"{time.perf_counter()-t0:.1f}s")

    # ── Dashboard ────────────────────────────────────────────────────────
    log("\n[bold]Building dashboard...[/bold]")
    app = build_dashboard(
        t_arr, V_arr, spikes_single,
        fi_currents, fi_rates,
        net_spikes, n_neurons, n_conns, total_spikes, mean_rate
    )

    log(f"\n  [bold green]Dashboard ready![/bold green]")
    log(f"  Opening http://127.0.0.1:{PORT}")
    log("  Press Ctrl+C to stop\n")

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)