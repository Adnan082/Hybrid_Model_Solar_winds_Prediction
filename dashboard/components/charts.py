"""dashboard/components/charts.py — All Plotly figure builders."""

from __future__ import annotations
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_TRANSPARENT = "rgba(0,0,0,0)"
_GRID        = "rgba(26,45,80,0.6)"
_FONT        = dict(family="Share Tech Mono, monospace", color="#8892b0", size=11)

_STORM_LINES = [
    (-30,  "#88ff44", "Minor"),
    (-50,  "#ffd700", "Moderate"),
    (-100, "#ff8c00", "Intense"),
    (-200, "#ff3366", "Extreme"),
]


def _base_layout(**kwargs) -> dict:
    return dict(
        paper_bgcolor = _TRANSPARENT,
        plot_bgcolor  = _TRANSPARENT,
        font          = _FONT,
        margin        = dict(l=48, r=16, t=36, b=36),
        legend        = dict(orientation="h", y=1.08, x=0,
                             font=dict(size=10), bgcolor=_TRANSPARENT),
        xaxis = dict(gridcolor=_GRID, zerolinecolor=_GRID, showgrid=True),
        yaxis = dict(gridcolor=_GRID, zerolinecolor=_GRID, showgrid=True),
        **kwargs,
    )


# ── 1. Main Dst time-series ───────────────────────────────────────────────────

def build_dst_chart(points: list[dict]) -> go.Figure:
    if not points:
        return _empty("No data yet — pipeline warming up...")

    times     = [p.get("timestamp", i) for i, p in enumerate(reversed(points))]
    dst_pred  = [p.get("dst_pred",      0.0) for p in reversed(points)]
    dst_burt  = [p.get("dst_burton",    0.0) for p in reversed(points)]
    dst_corr  = [p.get("dst_corrector", 0.0) for p in reversed(points)]

    fig = go.Figure()

    # Storm zone shading
    fig.add_hrect(y0=-30,  y1=0,   fillcolor="rgba(0,255,136,0.03)", line_width=0)
    fig.add_hrect(y0=-50,  y1=-30, fillcolor="rgba(255,215,0,0.04)", line_width=0)
    fig.add_hrect(y0=-100, y1=-50, fillcolor="rgba(255,140,0,0.05)", line_width=0)
    fig.add_hrect(y0=-300, y1=-100,fillcolor="rgba(255,51,102,0.06)",line_width=0)

    fig.add_trace(go.Scatter(
        x=times, y=dst_burt, name="Burton (physics)",
        line=dict(color="#ffd700", width=1, dash="dot"), opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=times, y=dst_corr, name="BiLSTM Corrector",
        line=dict(color="#bf5fff", width=1, dash="dash"), opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=times, y=dst_pred, name="RL Final",
        line=dict(color="#00d4ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
    ))

    for threshold, color, label in _STORM_LINES:
        fig.add_hline(y=threshold, line_color=color, line_dash="dot",
                      line_width=1, opacity=0.6,
                      annotation_text=label, annotation_font_size=9,
                      annotation_font_color=color, annotation_position="right")

    fig.update_layout(
        **_base_layout(title=dict(text="Dst Prediction", font=dict(size=13, color="#00d4ff"))),
        yaxis=dict(title="Dst (nT)", autorange="reversed",
                   gridcolor=_GRID, zerolinecolor=_GRID),
        xaxis=dict(title="Time", gridcolor=_GRID),
        hovermode="x unified",
    )
    return fig


# ── 2. Anomaly gauge ──────────────────────────────────────────────────────────

def build_anomaly_gauge(score: float, alert: str) -> go.Figure:
    color_map = {"GREEN": "#00ff88", "YELLOW": "#ffd700", "RED": "#ff3366"}
    needle_color = color_map.get(alert, "#00d4ff")

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = round(score, 3),
        delta = {"reference": 0.4, "valueformat": ".3f"},
        gauge = {
            "axis": {"range": [0, 1], "tickcolor": "#8892b0",
                     "tickfont": dict(size=10), "dtick": 0.2},
            "bar":  {"color": needle_color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0.0, 0.4], "color": "rgba(0,255,136,0.12)"},
                {"range": [0.4, 0.8], "color": "rgba(255,215,0,0.12)"},
                {"range": [0.8, 1.0], "color": "rgba(255,51,102,0.18)"},
            ],
            "threshold": {"line": {"color": "#ff3366", "width": 2},
                          "thickness": 0.75, "value": 0.8},
        },
        number = {"font": {"size": 32, "color": needle_color,
                           "family": "Share Tech Mono"}},
        title  = {"text": "Anomaly Score", "font": {"size": 12, "color": "#8892b0"}},
    ))
    fig.update_layout(
        paper_bgcolor=_TRANSPARENT, height=220,
        margin=dict(l=24, r=24, t=24, b=8),
        font=dict(family="Share Tech Mono", color="#8892b0"),
    )
    return fig


# ── 3. RL blend weights ───────────────────────────────────────────────────────

def build_blend_chart(points: list[dict]) -> go.Figure:
    if not points:
        return _empty("Waiting for RL data...")

    times  = [p.get("timestamp", i) for i, p in enumerate(reversed(points))]
    w_b    = [p.get("w_burton",    0.5) for p in reversed(points)]
    w_c    = [p.get("w_corrector", 0.5) for p in reversed(points)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=w_b, name="w_burton",
        stackgroup="one", line=dict(color="#ffd700", width=0),
        fillcolor="rgba(255,215,0,0.5)",
    ))
    fig.add_trace(go.Scatter(
        x=times, y=w_c, name="w_corrector",
        stackgroup="one", line=dict(color="#bf5fff", width=0),
        fillcolor="rgba(191,95,255,0.5)",
    ))
    fig.update_layout(
        **_base_layout(title=dict(text="RL Blend Weights", font=dict(size=12, color="#00d4ff"))),
        yaxis=dict(range=[0, 1], tickformat=".0%",
                   title="Weight", gridcolor=_GRID),
        hovermode="x unified",
    )
    return fig


# ── 4. Solar wind parameters ──────────────────────────────────────────────────

def build_solar_wind_chart(points: list[dict]) -> go.Figure:
    if not points:
        return _empty("No solar wind data yet...")

    times = [p.get("timestamp", i) for i, p in enumerate(reversed(points))]
    bz    = [p.get("bz_gsm",  0.0) for p in reversed(points)]
    speed = [p.get("speed", 400.0) for p in reversed(points)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=times, y=bz, name="Bz GSM (nT)",
        line=dict(color="#00d4ff", width=1.5),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=times, y=speed, name="Speed (km/s)",
        line=dict(color="#ff8c00", width=1.5),
    ), secondary_y=True)
    fig.add_hline(y=0, line_color="#4a5568", line_dash="dot", line_width=1)

    fig.update_layout(
        **_base_layout(title=dict(text="Solar Wind", font=dict(size=12, color="#00d4ff"))),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Bz (nT)",   gridcolor=_GRID, secondary_y=False)
    fig.update_yaxes(title_text="Speed (km/s)", gridcolor=_GRID, secondary_y=True,
                     showgrid=False)
    return fig


# ── 5. RL learning curve ──────────────────────────────────────────────────────

def build_rl_curve(points: list[dict]) -> go.Figure:
    if not points:
        return _empty("RL agent has not received rewards yet...")

    steps  = list(range(len(points)))
    cert   = [p.get("blend_certainty", 0.5) for p in reversed(points)]
    # rolling mean for smoothness
    window = 20
    smooth = [
        sum(cert[max(0, i-window):i+1]) / len(cert[max(0, i-window):i+1])
        for i in range(len(cert))
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=cert, name="Blend Certainty",
        line=dict(color="#4a5568", width=1), opacity=0.4,
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=smooth, name="Smoothed",
        line=dict(color="#00ff88", width=2),
    ))
    fig.update_layout(
        **_base_layout(title=dict(text="RL Learning Curve (Blend Certainty)", font=dict(size=12, color="#00d4ff"))),
        yaxis=dict(range=[0, 1], title="Certainty", gridcolor=_GRID),
        xaxis=dict(title="Prediction Step", gridcolor=_GRID),
    )
    return fig


# ── Helper ────────────────────────────────────────────────────────────────────

def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=13, color="#4a5568",
                                   family="Share Tech Mono"),
    )
    fig.update_layout(paper_bgcolor=_TRANSPARENT, plot_bgcolor=_TRANSPARENT,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=0, r=0, t=0, b=0))
    return fig
