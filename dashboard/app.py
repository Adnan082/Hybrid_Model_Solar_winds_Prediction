"""
dashboard/app.py
----------------
NASA-level real-time Solar Wind Dst Prediction Dashboard.

Run:
    python dashboard/app.py

Connects to FastAPI backend at API_BASE_URL (default: http://localhost:8000).
Refreshes every 3 seconds via dcc.Interval.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

from dashboard.components.charts import (
    build_dst_chart, build_anomaly_gauge, build_blend_chart,
    build_solar_wind_chart, build_rl_curve, _empty,
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
INTERVAL = 3_000   # ms

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Solar Wind DST | Mission Control",
    update_title=None,
    suppress_callback_exceptions=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch(path: str, params: dict | None = None) -> dict | list | None:
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params, timeout=2.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _kpi_card(card_id: str, label: str, unit: str = "") -> dbc.Col:
    return dbc.Col(
        html.Div([
            html.Div(label, className="kpi-label"),
            html.Div([
                html.Span("--", id=card_id, className="kpi-value"),
                html.Span(unit, className="kpi-unit"),
            ]),
        ], className="card", style={"textAlign": "center", "minHeight": "90px"}),
        width=3,
    )


def _agent_row(name: str, topic: str, status: str, idle: int) -> html.Div:
    dot_cls = f"agent-dot dot-{status}"
    return html.Div([
        html.Span(className=dot_cls),
        html.Span(name, style={"fontSize": "12px", "marginRight": "8px"}),
        html.Span(f"({topic})", style={"fontSize": "10px", "color": "#4a5568"}),
        html.Span(f"{idle}s ago" if idle >= 0 else "no data",
                  style={"fontSize": "10px", "color": "#8892b0", "float": "right"}),
    ], style={"padding": "6px 0", "borderBottom": "1px solid #1a2d50",
              "display": "flex", "alignItems": "center"})


# ── Layout ────────────────────────────────────────────────────────────────────

app.layout = html.Div([

    # ── Header ──────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div("SOLAR WIND DST PREDICTION SYSTEM", className="header-title"),
            html.Div("MULTI-AGENT ML PIPELINE  |  REAL-TIME MONITORING", className="header-subtitle"),
        ]),
        html.Div([
            html.Span(className="live-dot"),
            html.Span(id="header-clock", style={"fontFamily": "Share Tech Mono", "fontSize": "13px",
                                                  "color": "#8892b0", "marginRight": "16px"}),
            html.Span(id="header-alert", className="alert-badge alert-GREEN", children="GREEN"),
            html.Span("  "),
            html.Span(id="header-storm", className="storm-QUIET",
                      style={"fontFamily": "Share Tech Mono", "fontSize": "13px"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], className="header-bar"),

    # ── Body ─────────────────────────────────────────────────────────────────
    html.Div([

        # ── KPI row ──────────────────────────────────────────────────────────
        dbc.Row([
            _kpi_card("kpi-dst",       "PREDICTED Dst",   "nT"),
            _kpi_card("kpi-anomaly",   "ANOMALY SCORE",   ""),
            _kpi_card("kpi-confidence","CONFIDENCE",      "%"),
            _kpi_card("kpi-rl-steps",  "RL REWARD STEPS", ""),
        ], className="mb-3 g-2"),

        # ── Main charts row ───────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div("DST PREDICTION TIMELINE", className="section-title"),
                    dcc.Graph(id="chart-dst", config={"displayModeBar": False},
                              style={"height": "320px"}),
                ], className="card"),
                width=8,
            ),
            dbc.Col(
                html.Div([
                    html.Div("ANOMALY DETECTOR", className="section-title"),
                    dcc.Graph(id="chart-gauge", config={"displayModeBar": False},
                              style={"height": "220px"}),
                    html.Div(id="alert-detail",
                             style={"textAlign": "center", "marginTop": "8px",
                                    "fontSize": "11px", "color": "#8892b0"}),
                ], className="card"),
                width=4,
            ),
        ], className="mb-3 g-2"),

        # ── Secondary row ─────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div("RL BLEND WEIGHTS", className="section-title"),
                    dcc.Graph(id="chart-blend", config={"displayModeBar": False},
                              style={"height": "220px"}),
                ], className="card"),
                width=4,
            ),
            dbc.Col(
                html.Div([
                    html.Div("SOLAR WIND PARAMETERS", className="section-title"),
                    dcc.Graph(id="chart-solar-wind", config={"displayModeBar": False},
                              style={"height": "220px"}),
                ], className="card"),
                width=4,
            ),
            dbc.Col(
                html.Div([
                    html.Div("RL LEARNING CURVE", className="section-title"),
                    dcc.Graph(id="chart-rl-curve", config={"displayModeBar": False},
                              style={"height": "220px"}),
                ], className="card"),
                width=4,
            ),
        ], className="mb-3 g-2"),

        # ── Bottom row ────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div("AGENT PIPELINE STATUS", className="section-title"),
                    html.Div(id="agent-status-panel"),
                ], className="card"),
                width=6,
            ),
            dbc.Col(
                html.Div([
                    html.Div("PREDICTION BREAKDOWN", className="section-title"),
                    html.Div(id="prediction-breakdown"),
                ], className="card"),
                width=6,
            ),
        ], className="mb-3 g-2"),

    ], style={"padding": "16px"}),

    # ── Interval timer ────────────────────────────────────────────────────────
    dcc.Interval(id="main-interval", interval=INTERVAL, n_intervals=0),
    dcc.Store(id="store-latest"),
    dcc.Store(id="store-history"),
], style={"minHeight": "100vh"})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("store-latest",  "data"),
    Output("store-history", "data"),
    Input("main-interval",  "n_intervals"),
)
def fetch_data(_n):
    latest  = _fetch("/api/v1/prediction/latest") or {}
    history = _fetch("/api/v1/history/predictions", {"n": 120})
    points  = history.get("points", []) if history else []
    return latest, points


@app.callback(
    Output("header-clock",  "children"),
    Output("header-alert",  "children"),
    Output("header-alert",  "className"),
    Output("header-storm",  "children"),
    Output("header-storm",  "className"),
    Output("kpi-dst",       "children"),
    Output("kpi-dst",       "style"),
    Output("kpi-anomaly",   "children"),
    Output("kpi-confidence","children"),
    Output("kpi-rl-steps",  "children"),
    Input("store-latest",   "data"),
    Input("main-interval",  "n_intervals"),
)
def update_header_kpis(latest, _n):
    from datetime import datetime, timezone
    clock = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    if not latest:
        return (clock, "NO DATA", "alert-badge alert-GREEN",
                "QUIET", "storm-QUIET",
                "--", {}, "--", "--", "--")

    dst     = latest.get("dst_pred",      0.0)
    alert   = latest.get("alert_level",   "GREEN")
    anomaly = latest.get("anomaly_score", 0.0)
    conf    = latest.get("confidence",    1.0)
    steps   = latest.get("rl_reward_steps", 0)

    color_map = {"GREEN": "#00ff88", "YELLOW": "#ffd700", "RED": "#ff3366"}
    storm_map = {
        "QUIET": "QUIET", "MINOR": "MINOR STORM", "MODERATE": "MODERATE STORM",
        "INTENSE": "INTENSE STORM", "EXTREME": "EXTREME STORM",
    }
    def classify(d):
        if d > -30:   return "QUIET"
        if d > -50:   return "MINOR"
        if d > -100:  return "MODERATE"
        if d > -200:  return "INTENSE"
        return "EXTREME"

    storm_cls = classify(dst)

    return (
        clock,
        alert, f"alert-badge alert-{alert}",
        storm_map.get(storm_cls, storm_cls), f"storm-{storm_cls}",
        f"{dst:.1f}", {"color": color_map.get(alert, "#00d4ff")},
        f"{anomaly:.3f}",
        f"{conf*100:.1f}",
        f"{steps:,}",
    )


@app.callback(
    Output("chart-dst",        "figure"),
    Output("chart-blend",      "figure"),
    Output("chart-solar-wind", "figure"),
    Output("chart-rl-curve",   "figure"),
    Input("store-history",     "data"),
)
def update_charts(points):
    pts = points or []
    return (
        build_dst_chart(pts),
        build_blend_chart(pts),
        build_solar_wind_chart(pts),
        build_rl_curve(pts),
    )


@app.callback(
    Output("chart-gauge",  "figure"),
    Output("alert-detail", "children"),
    Input("store-latest",  "data"),
)
def update_gauge(latest):
    if not latest:
        return _empty("Waiting..."), "No data"
    score = latest.get("anomaly_score", 0.0)
    alert = latest.get("alert_level",   "GREEN")
    desc  = {
        "GREEN":  "Quiet solar wind — Burton physics reliable",
        "YELLOW": "Storm onset likely — monitor closely",
        "RED":    "Active storm — ML correction engaged",
    }.get(alert, "")
    return build_anomaly_gauge(score, alert), desc


@app.callback(
    Output("agent-status-panel", "children"),
    Input("main-interval", "n_intervals"),
)
def update_agents(_n):
    data = _fetch("/api/v1/agents/status") or []
    if not data:
        return html.Div("Cannot reach API", style={"color": "#4a5568", "fontSize": "12px"})
    return [_agent_row(a["name"], a["topic"], a["status"], a["idle_secs"]) for a in data]


@app.callback(
    Output("prediction-breakdown", "children"),
    Input("store-latest", "data"),
)
def update_breakdown(latest):
    if not latest:
        return html.Div("No prediction yet", style={"color": "#4a5568"})

    rows = [
        ("Burton (Physics)",  f"{latest.get('dst_burton',    0.0):>8.2f} nT", "#ffd700"),
        ("BiLSTM Corrector",  f"{latest.get('dst_corrector', 0.0):>8.2f} nT", "#bf5fff"),
        ("RL Final Blend",    f"{latest.get('dst_pred',       0.0):>8.2f} nT", "#00d4ff"),
        ("Residual Pred",     f"{latest.get('residual_pred',  0.0):>8.2f} nT", "#8892b0"),
        ("w_burton",          f"{latest.get('w_burton',       0.5)*100:>7.1f} %",  "#ffd700"),
        ("w_corrector",       f"{latest.get('w_corrector',    0.5)*100:>7.1f} %",  "#bf5fff"),
        ("Blend Certainty",   f"{latest.get('blend_certainty',0.0)*100:>7.1f} %",  "#00ff88"),
        ("Storm Phase",       str(latest.get("storm_phase", 0)),               "#8892b0"),
    ]
    return html.Div([
        html.Div([
            html.Span(label, style={"fontSize": "11px", "color": "#8892b0",
                                     "fontFamily": "Share Tech Mono", "width": "160px",
                                     "display": "inline-block"}),
            html.Span(value, style={"fontSize": "13px", "color": color,
                                     "fontFamily": "Share Tech Mono", "fontWeight": "600"}),
        ], style={"padding": "5px 0", "borderBottom": "1px solid #1a2d50"})
        for label, value, color in rows
    ])


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
