"""Interactive Streamlit dashboard for the Sampling and Aliasing Visualizer.

Implements the visual design, layout, presets, synchronized controls,
time-domain visualization, audio comparison studio, and 4-step DSP
interpretation panel defined in `Sampling & Aliasing Visualizer.html`
on top of the core DSP engine in `dsp_engine.py`.
"""

from __future__ import annotations

import html
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dsp_engine import (
    PRESETS,
    SUPPORTED_WAVEFORMS,
    UI_COLORS,
    _format_num,
    analyze_sampling,
    simulate_sampling,
    synthesize_audio_pair,
)

DEFAULT_PRESET_ID = "classic_undersampling"
DEFAULT_SCALE_FACTOR = 50.0


# ---------------------------------------------------------------------------
# Custom CSS & Pure HTML Builders
# ---------------------------------------------------------------------------


def inject_custom_css() -> None:
    """Inject Inter typography and design-system styling into the Streamlit page."""
    c = UI_COLORS
    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    color: {c["text_primary"]};
}}

.stApp {{
    background-color: {c["bg_app"]};
}}

[data-testid="stHeader"] {{
    background-color: {c["bg_app"]};
    color: {c["text_primary"]};
}}

[data-testid="stSidebar"] {{
    background-color: {c["bg_surface"]};
    border-right: 1px solid {c["border_subtle"]};
    color: {c["text_primary"]};
}}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0.55rem;
}}

[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stCheckbox label p,
.stCheckbox label,
.stCheckbox label p {{
    color: {c["text_primary"]} !important;
    font-size: 13px !important;
}}

[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
[data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] {{
    display: none !important;
}}

[data-testid="stSidebar"] input[type="number"] {{
    background-color: {c["bg_surface"]} !important;
    color: {c["text_primary"]} !important;
}}

[data-testid="stSidebar"] [data-testid="stNumberInputContainer"] {{
    border: 1.5px solid {c["border_control"]} !important;
    border-radius: 6px !important;
    background-color: {c["bg_surface"]} !important;
    min-height: 36px !important;
}}

[data-testid="stSidebar"] [data-testid="stNumberInputContainer"]:focus-within {{
    border-color: {c["accent_primary"]} !important;
}}

/* Visibility checkbox color swatches */
.st-key-show_continuous [data-testid="stWidgetLabel"] p::before,
.st-key-show_samples [data-testid="stWidgetLabel"] p::before,
.st-key-show_recon [data-testid="stWidgetLabel"] p::before,
.st-key-show_alias_overlay [data-testid="stWidgetLabel"] p::before {{
    content: "";
    display: inline-block;
    width: 11px;
    height: 11px;
    border-radius: 3px;
    margin-right: 7px;
    vertical-align: -1px;
}}

.st-key-show_continuous [data-testid="stWidgetLabel"] p::before {{
    background-color: {c["trace_continuous"]};
}}

.st-key-show_samples [data-testid="stWidgetLabel"] p::before {{
    background-color: {c["trace_samples"]};
}}

.st-key-show_recon [data-testid="stWidgetLabel"] p::before {{
    background-color: {c["trace_reconstructed"]};
}}

.st-key-show_alias_overlay [data-testid="stWidgetLabel"] p::before {{
    background-color: #D97706;
}}

.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1220px;
}}

.sav-sidebar-kicker {{
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: {c["text_secondary"]};
    text-transform: uppercase;
    margin-bottom: 2px;
}}

.sav-sidebar-heading {{
    font-size: 13px;
    font-weight: 600;
    color: {c["text_primary"]};
    margin-top: 8px;
    margin-bottom: 2px;
}}

.sav-param-label {{
    font-size: 13px;
    font-weight: 600;
    color: {c["text_primary"]};
    margin-top: 6px;
    margin-bottom: 4px;
}}

/* Preset and waveform sidebar buttons */
[data-testid="stSidebar"] .stButton > button {{
    border-radius: 8px;
    min-height: 42px;
    font-size: 13px;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}}

[data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
    background-color: {c["bg_surface"]};
    border: 1.5px solid {c["border_control"]};
    color: {c["text_primary"]};
}}

[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {{
    border-color: {c["accent_primary"]};
    background-color: #F7F9FE;
}}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background-color: {c["accent_soft"]};
    border: 1.5px solid {c["accent_primary"]};
    color: {c["text_primary"]};
}}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
    background-color: #DCE6FC;
    border-color: {c["accent_primary"]};
}}

/* Left-align multi-line preset buttons cleanly */
.st-key-sav_presets .stButton > button {{
    text-align: left !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
    padding: 9px 12px !important;
    line-height: 1.35 !important;
}}

.st-key-sav_presets .stButton > button > div,
.st-key-sav_presets .stButton > button [data-testid="stMarkdownContainer"] {{
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
}}

.st-key-sav_presets .stButton > button [data-testid="stMarkdownContainer"] p {{
    text-align: left !important;
    width: 100%;
    margin: 0;
}}

.st-key-sav_presets .stButton > button [data-testid="stMarkdownContainer"] p:first-child {{
    font-weight: 600;
    font-size: 13px;
    color: {c["text_primary"]};
}}

.st-key-sav_presets .stButton > button [data-testid="stMarkdownContainer"] p:last-child {{
    font-weight: 400;
    font-size: 12px;
    color: {c["text_secondary"]};
    margin-top: 2px;
}}

@media (min-width: 768px) {{
    [data-testid="stSidebar"][aria-expanded="true"] {{
        min-width: 292px !important;
        width: 292px !important;
    }}
}}

/* Active waveform pill matches dark pill in HTML design */
.st-key-sav_waveforms [data-testid="stHorizontalBlock"] {{
    gap: 6px !important;
}}

.st-key-sav_waveforms .stButton > button {{
    padding: 0 4px !important;
    font-size: 12.5px !important;
    white-space: nowrap !important;
}}

.st-key-sav_waveforms .stButton > button p {{
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
}}

.st-key-sav_waveforms .stButton > button[kind="primary"] {{
    background-color: {c["text_primary"]} !important;
    border-color: {c["text_primary"]} !important;
    color: #FFFFFF !important;
}}

.st-key-sav_waveforms .stButton > button[kind="primary"] p {{
    color: #FFFFFF !important;
}}

.st-key-sav_waveforms .stButton > button[kind="secondary"] {{
    border-color: {c["text_primary"]} !important;
    color: {c["text_primary"]} !important;
}}

.sav-header h1 {{
    margin: 0;
    font-size: 28px;
    font-weight: 700;
    line-height: 1.2;
    color: {c["text_primary"]};
}}

.sav-header p {{
    margin: 4px 0 0 0;
    font-size: 14px;
    color: {c["text_secondary"]};
}}

.sav-banner {{
    border-radius: 10px;
    padding: 14px 16px;
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    margin-top: 14px;
    margin-bottom: 14px;
}}

.sav-banner-badge {{
    color: #FFFFFF;
    font-size: 12px;
    font-weight: 700;
    border-radius: 6px;
    padding: 4px 10px;
    letter-spacing: 0.02em;
}}

.sav-banner-msg {{
    font-size: 14px;
    font-weight: 600;
}}

.sav-kpi-grid {{
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}}

@media (max-width: 1100px) {{
    .sav-kpi-grid {{
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
}}

@media (max-width: 640px) {{
    .sav-kpi-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
}}

.sav-kpi-card {{
    background-color: {c["bg_surface"]};
    border: 1px solid {c["border_subtle"]};
    border-radius: 10px;
    padding: 13px 14px;
}}

.sav-kpi-label {{
    font-size: 12px;
    color: {c["text_secondary"]};
}}

.sav-kpi-value {{
    font-size: 20px;
    font-weight: 700;
    margin-top: 4px;
    color: {c["text_primary"]};
    line-height: 1.2;
    white-space: nowrap;
    letter-spacing: -0.01em;
}}

.sav-kpi-sub {{
    font-size: 12px;
    color: {c["text_secondary"]};
    margin-top: 2px;
}}

.sav-audio-pill {{
    display: inline-block;
    margin-top: 5px;
    padding: 2px 8px;
    border-radius: 5px;
    background-color: {c["bg_app"]};
    border: 1px solid {c["border_subtle"]};
    font-size: 11.5px;
    color: {c["text_secondary"]};
    font-variant-numeric: tabular-nums;
}}

.sav-card-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 6px;
}}

.sav-card-title {{
    margin: 0;
    font-size: 16px;
    font-weight: 700;
    color: {c["text_primary"]};
    word-break: normal;
    overflow-wrap: normal;
}}

.sav-legend {{
    display: flex;
    gap: 16px;
    font-size: 12px;
    color: {c["text_secondary"]};
    flex-wrap: wrap;
}}

.sav-audio-meta {{
    margin-bottom: 6px;
}}

.sav-audio-title {{
    font-weight: 600;
    font-size: 14px;
    color: {c["text_primary"]};
}}

.sav-audio-sub {{
    font-size: 12px;
    color: {c["text_secondary"]};
}}

.sav-steps {{
    display: flex;
    flex-direction: column;
    gap: 14px;
    margin-top: 8px;
}}

.sav-step-row {{
    display: flex;
    gap: 12px;
    align-items: flex-start;
}}

.sav-step-badge {{
    flex: none;
    width: 26px;
    height: 26px;
    border-radius: 13px;
    background-color: {c["text_primary"]};
    color: #FFFFFF;
    font-size: 13px;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;
}}

.sav-step-title {{
    font-weight: 600;
    font-size: 14px;
    color: {c["text_primary"]};
}}

.sav-step-body {{
    font-size: 13px;
    color: {c["text_secondary"]};
    margin-top: 2px;
    line-height: 1.45;
}}

.sav-harmonics-box {{
    margin-top: 12px;
    padding: 10px 12px;
    border-radius: 8px;
    background-color: {c["bg_app"]};
    border: 1px solid {c["border_subtle"]};
    font-size: 12px;
    color: {c["text_secondary"]};
}}

.sav-harmonics-pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 6px;
}}

.sav-harmonic-pill {{
    background-color: {c["bg_surface"]};
    border: 1px solid {c["border_subtle"]};
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 11.5px;
    color: {c["text_primary"]};
}}

/* Give bordered Streamlit containers the clean white card surface */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {c["bg_surface"]};
    border-color: {c["border_subtle"]} !important;
    border-radius: 10px !important;
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


def render_header_html() -> str:
    """Build the dashboard header HTML block."""
    return (
        '<div class="sav-header">'
        "<h1>Sampling and Aliasing Visualizer</h1>"
        "<p>Vary the signal and sampling frequency and watch the sampled waveform change.</p>"
        "</div>"
    )


def render_status_banner_html(analysis: dict[str, Any]) -> str:
    """Build the color-coded Nyquist/aliasing status banner HTML."""
    level = analysis["banner_level"]
    if level == "danger":
        bg = UI_COLORS["status_danger_bg"]
        border = UI_COLORS["status_danger_border"]
        badge_bg = UI_COLORS["status_danger_badge"]
        text_color = UI_COLORS["status_danger_text"]
    elif level == "warning":
        bg = UI_COLORS["status_warning_bg"]
        border = UI_COLORS["status_warning_border"]
        badge_bg = UI_COLORS["status_warning_badge"]
        text_color = UI_COLORS["status_warning_text"]
    else:
        bg = UI_COLORS["status_safe_bg"]
        border = UI_COLORS["status_safe_border"]
        badge_bg = UI_COLORS["status_safe_badge"]
        text_color = UI_COLORS["status_safe_text"]

    badge = html.escape(str(analysis["badge_text"]))
    msg = html.escape(str(analysis["banner_message"]))
    return (
        f'<div class="sav-banner" style="background:{bg};border:1.5px solid {border};">'
        f'<span class="sav-banner-badge" style="background:{badge_bg};">{badge}</span>'
        f'<span class="sav-banner-msg" style="color:{text_color};">{msg}</span>'
        f"</div>"
    )


def render_kpi_grid_html(kpi_cards: list[dict[str, str]]) -> str:
    """Build the 6-card responsive KPI grid HTML."""
    cards_html = []
    for card in kpi_cards:
        label = html.escape(str(card["label"]))
        value = html.escape(str(card["value"]))
        subtitle = html.escape(str(card["subtitle"]))
        cards_html.append(
            '<div class="sav-kpi-card">'
            f'<div class="sav-kpi-label">{label}</div>'
            f'<div class="sav-kpi-value">{value}</div>'
            f'<div class="sav-kpi-sub">{subtitle}</div>'
            "</div>"
        )
    return f'<div class="sav-kpi-grid">{"".join(cards_html)}</div>'


def render_time_domain_header_html() -> str:
    """Build the Time-domain view card header with trace legend."""
    c_cont = UI_COLORS["trace_continuous"]
    c_samp = UI_COLORS["trace_samples"]
    c_rec = UI_COLORS["trace_reconstructed"]
    return (
        '<div class="sav-card-head">'
        '<h2 class="sav-card-title">Time-domain view</h2>'
        '<div class="sav-legend">'
        f'<span><b style="color:{c_cont};">&#9473;</b> x(t)</span>'
        f'<span><b style="color:{c_samp};">&#9679;</b> x[n]</span>'
        f'<span><b style="color:{c_rec};">&#9549;</b> x&#770;(t), alias</span>'
        "</div>"
        "</div>"
    )


def render_interpretation_panel_html(
    analysis: dict[str, Any], waveform: str
) -> str:
    """Build the 4-step DSP interpretation HTML plus harmonic folding pills if applicable."""
    rows = []
    for step in analysis["steps"]:
        num = html.escape(str(step["number"]))
        title = html.escape(str(step["title"]))
        body = html.escape(str(step["body"]))
        rows.append(
            '<div class="sav-step-row">'
            f'<div class="sav-step-badge">{num}</div>'
            "<div>"
            f'<div class="sav-step-title">{title}</div>'
            f'<div class="sav-step-body">{body}</div>'
            "</div>"
            "</div>"
        )

    harmonics_html = ""
    harmonic_aliases = analysis.get("harmonic_aliases", [])
    if waveform != "Sine" and harmonic_aliases:
        pills = []
        for h in harmonic_aliases[:5]:
            order = h["order"]
            hf = _format_num(h["harmonic_freq"])
            af = _format_num(h["alias_freq"])
            inv = " (inv)" if h["phase_inverted"] else ""
            pills.append(
                f'<span class="sav-harmonic-pill">{order}f&#8320;: {hf} Hz &rarr; {af} Hz{inv}</span>'
            )
        harmonics_html = (
            '<div class="sav-harmonics-box">'
            "<strong>Folded odd harmonics (above fs/2):</strong>"
            f'<div class="sav-harmonics-pills">{"".join(pills)}</div>'
            "</div>"
        )

    return f'<div class="sav-steps">{"".join(rows)}</div>{harmonics_html}'


# ---------------------------------------------------------------------------
# Interactive Plotly Time-Domain Figure Builder
# ---------------------------------------------------------------------------


def build_time_domain_figure(
    sim: dict[str, Any],
    show_continuous: bool = True,
    show_samples: bool = True,
    show_recon: bool = True,
    show_alias_overlay: bool = True,
) -> go.Figure:
    """Build the interactive Plotly time-domain chart overlaying x(t), x[n], and x̂(t)."""
    t_cont = sim["t_cont"]
    x_cont = sim["x_cont"]
    t_samples = sim["t_samples"]
    x_samples = sim["x_samples"]
    x_recon = sim["x_recon"]
    x_alias = sim["x_alias_fundamental"]

    fig = go.Figure()

    if show_continuous:
        fig.add_trace(
            go.Scatter(
                x=t_cont,
                y=x_cont,
                mode="lines",
                name="Continuous x(t)",
                line=dict(color=UI_COLORS["trace_continuous"], width=2.0),
                opacity=0.82,
                hovertemplate="t = %{x:.4f} s<br>x(t) = %{y:.3f}<extra>x(t)</extra>",
            )
        )

    if show_recon:
        fig.add_trace(
            go.Scatter(
                x=t_cont,
                y=x_recon,
                mode="lines",
                name="Reconstruction x̂(t)",
                line=dict(
                    color=UI_COLORS["trace_reconstructed"],
                    width=2.6,
                    dash="dash",
                ),
                hovertemplate="t = %{x:.4f} s<br>x̂(t) = %{y:.3f}<extra>x̂(t)</extra>",
            )
        )

    if show_alias_overlay:
        if show_recon and sim.get("waveform") == "Sine":
            alias_color = "rgba(194, 65, 12, 0.22)"
            alias_dash = "solid"
            alias_width = 7.0
        elif show_recon:
            alias_color = "#D97706"
            alias_dash = "dot"
            alias_width = 2.2
        else:
            alias_color = UI_COLORS["trace_reconstructed"]
            alias_dash = "dash"
            alias_width = 2.5
        fig.add_trace(
            go.Scatter(
                x=t_cont,
                y=x_alias,
                mode="lines",
                name="Fundamental alias",
                line=dict(color=alias_color, width=alias_width, dash=alias_dash),
                hovertemplate="t = %{x:.4f} s<br>alias = %{y:.3f}<extra>Fundamental alias</extra>",
            )
        )

    if show_samples and len(t_samples) > 0:
        stem_x: list[float | None] = []
        stem_y: list[float | None] = []
        for ts, xs in zip(t_samples, x_samples):
            stem_x.extend([float(ts), float(ts), None])
            stem_y.extend([0.0, float(xs), None])

        fig.add_trace(
            go.Scatter(
                x=stem_x,
                y=stem_y,
                mode="lines",
                name="Sample stems",
                line=dict(color=UI_COLORS["trace_samples"], width=2.0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t_samples,
                y=x_samples,
                mode="markers",
                name="Samples x[n]",
                marker=dict(
                    color=UI_COLORS["trace_samples"],
                    size=9,
                    line=dict(color="#FFFFFF", width=1.2),
                ),
                hovertemplate="t[n] = %{x:.4f} s<br>x[n] = %{y:.3f}<extra>x[n]</extra>",
            )
        )

    peak_amp = max(float(sim["amplitude"]), 0.25)
    if show_recon and x_recon.size > 0:
        peak_amp = max(peak_amp, float(np.max(np.abs(x_recon))))
    y_bound = 1.2 * peak_amp

    fig.add_hline(y=0.0, line_color=UI_COLORS["border_subtle"], line_width=1.2)
    fig.update_layout(
        height=350,
        margin=dict(l=60, r=18, t=12, b=44),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        showlegend=False,
        hovermode="closest",
        font=dict(
            family="Inter, system-ui, sans-serif",
            size=12,
            color=UI_COLORS["text_secondary"],
        ),
        xaxis=dict(
            title="t (s)",
            range=[0.0, float(sim["duration"])],
            showgrid=True,
            gridcolor="#ECEFF4",
            zeroline=False,
            linecolor=UI_COLORS["border_subtle"],
            automargin=True,
        ),
        yaxis=dict(
            title="Amplitude",
            range=[-y_bound, y_bound],
            showgrid=True,
            gridcolor="#ECEFF4",
            zeroline=False,
            linecolor=UI_COLORS["border_subtle"],
            automargin=True,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Cached DSP Wrappers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_simulate_and_analyze(
    waveform: str,
    f0: float,
    fs: float,
    amplitude: float,
    phase_deg: float,
    duration: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run and cache simulation + DSP analysis for identical parameter tuples."""
    sim = simulate_sampling(
        waveform=waveform,
        f0=f0,
        fs=fs,
        amplitude=amplitude,
        phase_deg=phase_deg,
        duration=duration,
    )
    analysis = analyze_sampling(sim)
    return sim, analysis


@st.cache_data(show_spinner=False)
def _cached_synthesize_audio(
    waveform: str,
    f0: float,
    fs: float,
    amplitude: float,
    phase_deg: float,
    audible_scaling: bool,
    scale_factor: float,
) -> dict[str, Any]:
    """Synthesize and cache WAV audio streams for the comparison studio."""
    return synthesize_audio_pair(
        waveform=waveform,
        f0=f0,
        fs=fs,
        amplitude=amplitude,
        phase_deg=phase_deg,
        audible_scaling=audible_scaling,
        scale_factor=scale_factor,
    )


# ---------------------------------------------------------------------------
# Session State & Bidirectional Synchronization Callbacks
# ---------------------------------------------------------------------------


_PARAM_WIDGET_MAP = {
    "f0": ("f0_slider", "f0_num"),
    "fs": ("fs_slider", "fs_num"),
    "amplitude": ("amp_slider", "amp_num"),
    "duration": ("dur_slider", "dur_num"),
}


def apply_preset(preset_id: str) -> None:
    """Apply an educational preset and sync all canonical & widget session keys."""
    cfg = PRESETS[preset_id]
    st.session_state["active_preset"] = preset_id
    st.session_state["waveform"] = cfg["waveform"]
    st.session_state["f0"] = float(cfg["f0"])
    st.session_state["fs"] = float(cfg["fs"])
    st.session_state["amplitude"] = float(cfg["amplitude"])
    st.session_state["duration"] = float(cfg["duration"])
    st.session_state["audible_scaling"] = bool(cfg["audible_scaling"])
    st.session_state["scale_factor"] = (
        float(cfg["scale_factor"]) if cfg["scale_factor"] > 1.0 else DEFAULT_SCALE_FACTOR
    )

    for canonical, (s_key, n_key) in _PARAM_WIDGET_MAP.items():
        val = float(st.session_state[canonical])
        st.session_state[s_key] = val
        st.session_state[n_key] = val


def init_session_state() -> None:
    """Initialize session state with the Classic Undersampling preset on first load."""
    if "initialized" not in st.session_state:
        st.session_state["initialized"] = True
        st.session_state["show_continuous"] = True
        st.session_state["show_samples"] = True
        st.session_state["show_recon"] = True
        st.session_state["show_alias_overlay"] = True
        apply_preset(DEFAULT_PRESET_ID)


def _on_slider_change(canonical: str) -> None:
    s_key, n_key = _PARAM_WIDGET_MAP[canonical]
    val = float(st.session_state[s_key])
    st.session_state[canonical] = val
    st.session_state[n_key] = val
    st.session_state["active_preset"] = "custom"


def _on_number_change(canonical: str) -> None:
    s_key, n_key = _PARAM_WIDGET_MAP[canonical]
    raw = st.session_state[n_key]
    if raw is None:
        return
    val = float(raw)
    st.session_state[canonical] = val
    st.session_state[s_key] = val
    st.session_state["active_preset"] = "custom"


def _set_waveform(wf: str) -> None:
    st.session_state["waveform"] = wf
    st.session_state["active_preset"] = "custom"


def _is_high_freq_range() -> bool:
    """Expand slider bounds dynamically when high-frequency values are active."""
    return (
        st.session_state["f0"] > 100.0
        or st.session_state["fs"] > 200.0
        or st.session_state["duration"] < 0.05
    )


# ---------------------------------------------------------------------------
# Sidebar & Main Dashboard Rendering
# ---------------------------------------------------------------------------


def _render_synced_param(
    label: str,
    canonical: str,
    slider_min: float,
    slider_max: float,
    num_min: float,
    num_max: float,
    step: float,
    fmt: str,
) -> None:
    """Render a label followed by a synchronized slider and numeric input pair."""
    s_key, n_key = _PARAM_WIDGET_MAP[canonical]
    st.markdown(f'<div class="sav-param-label">{label}</div>', unsafe_allow_html=True)
    col_s, col_n = st.columns([2.1, 1.0], gap="small")
    with col_s:
        st.slider(
            label=f"{label} slider",
            min_value=float(slider_min),
            max_value=float(slider_max),
            step=float(step),
            format=fmt,
            key=s_key,
            label_visibility="collapsed",
            on_change=_on_slider_change,
            args=(canonical,),
        )
    with col_n:
        st.number_input(
            label=f"{label} value",
            min_value=float(num_min),
            max_value=float(num_max),
            step=float(step),
            format=fmt,
            key=n_key,
            label_visibility="collapsed",
            on_change=_on_number_change,
            args=(canonical,),
        )


def render_sidebar() -> None:
    """Render the Control Center sidebar matching Sampling & Aliasing Visualizer.html."""
    with st.sidebar:
        st.markdown('<div class="sav-sidebar-kicker">CONTROL CENTER</div>', unsafe_allow_html=True)

        # Educational Presets
        st.markdown('<div class="sav-sidebar-heading">Educational presets</div>', unsafe_allow_html=True)
        active_preset = st.session_state["active_preset"]
        with st.container(key="sav_presets"):
            for preset_id, cfg in PRESETS.items():
                is_active = active_preset == preset_id
                btn_label = f"**{cfg['title']}**\n\n{cfg['subtitle']}"
                st.button(
                    btn_label,
                    key=f"preset_{preset_id}",
                    type="primary" if is_active else "secondary",
                    width="stretch",
                    on_click=apply_preset,
                    args=(preset_id,),
                )

        # Waveform Selector
        st.markdown('<div class="sav-sidebar-heading">Waveform</div>', unsafe_allow_html=True)
        with st.container(key="sav_waveforms"):
            wf_cols = st.columns([0.85, 1.0, 1.15], gap="small")
            current_wf = st.session_state["waveform"]
            for col, wf in zip(wf_cols, SUPPORTED_WAVEFORMS):
                with col:
                    st.button(
                        wf,
                        key=f"wf_{wf.lower()}",
                        type="primary" if current_wf == wf else "secondary",
                        width="stretch",
                        on_click=_set_waveform,
                        args=(wf,),
                    )

        # Parameter Controls (with dynamic high-freq range adaptation)
        high_freq = _is_high_freq_range()
        f0_max = 1200.0 if high_freq else 100.0
        fs_max = 2400.0 if high_freq else 200.0
        dur_min = 0.005 if high_freq else 0.05
        dur_step = 0.005 if high_freq else 0.05
        dur_fmt = "%.3f" if high_freq else "%.2f"

        _render_synced_param(
            label="Signal frequency f₀ (Hz)",
            canonical="f0",
            slider_min=1.0,
            slider_max=f0_max,
            num_min=1.0,
            num_max=1200.0,
            step=1.0,
            fmt="%.0f",
        )
        _render_synced_param(
            label="Sampling frequency fs (Hz)",
            canonical="fs",
            slider_min=1.0,
            slider_max=fs_max,
            num_min=1.0,
            num_max=2400.0,
            step=1.0,
            fmt="%.0f",
        )
        _render_synced_param(
            label="Amplitude A",
            canonical="amplitude",
            slider_min=0.1,
            slider_max=2.0,
            num_min=0.1,
            num_max=2.0,
            step=0.1,
            fmt="%.1f",
        )
        _render_synced_param(
            label="Time window T (s)",
            canonical="duration",
            slider_min=dur_min,
            slider_max=2.0,
            num_min=0.005,
            num_max=2.0,
            step=dur_step,
            fmt=dur_fmt,
        )

        # Visibility Toggles
        st.markdown('<div class="sav-sidebar-heading">Visibility</div>', unsafe_allow_html=True)
        st.checkbox("Continuous x(t)", key="show_continuous")
        st.checkbox("Samples x[n]", key="show_samples")
        st.checkbox("Reconstruction x̂(t)", key="show_recon")
        st.checkbox("Fundamental alias overlay", key="show_alias_overlay")


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(
        page_title="Sampling and Aliasing Visualizer",
        page_icon="〰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    inject_custom_css()
    render_sidebar()

    waveform = st.session_state["waveform"]
    f0 = float(st.session_state["f0"])
    fs = float(st.session_state["fs"])
    amplitude = float(st.session_state["amplitude"])
    phase_deg = 0.0
    duration = float(st.session_state["duration"])

    sim, analysis = _cached_simulate_and_analyze(
        waveform=waveform,
        f0=f0,
        fs=fs,
        amplitude=amplitude,
        phase_deg=phase_deg,
        duration=duration,
    )

    # Header + Status Banner + 6 KPI Cards
    st.markdown(render_header_html(), unsafe_allow_html=True)
    st.markdown(render_status_banner_html(analysis), unsafe_allow_html=True)
    st.markdown(render_kpi_grid_html(analysis["kpi_cards"]), unsafe_allow_html=True)

    # Interactive Time-Domain Card
    with st.container(border=True):
        st.markdown(render_time_domain_header_html(), unsafe_allow_html=True)
        fig = build_time_domain_figure(
            sim,
            show_continuous=bool(st.session_state["show_continuous"]),
            show_samples=bool(st.session_state["show_samples"]),
            show_recon=bool(st.session_state["show_recon"]),
            show_alias_overlay=bool(st.session_state["show_alias_overlay"]),
        )
        st.plotly_chart(
            fig,
            width="stretch",
            theme=None,
            config={"displayModeBar": "hover", "displaylogo": False},
        )

    # Two-Column Lower Section: Audio Comparison Studio & DSP Interpretation
    col_audio, col_interp = st.columns(2, gap="medium")

    with col_audio:
        with st.container(border=True):
            st.markdown('<h2 class="sav-card-title">Audio comparison studio</h2>', unsafe_allow_html=True)

            scale_factor = float(st.session_state.get("scale_factor", DEFAULT_SCALE_FACTOR))
            if scale_factor <= 1.0:
                scale_factor = DEFAULT_SCALE_FACTOR
                st.session_state["scale_factor"] = scale_factor

            is_scaled = bool(st.session_state["audible_scaling"])
            if is_scaled:
                scale_label = (
                    f"Audible scaling ×{_format_num(scale_factor)} "
                    f"({_format_num(f0)} Hz → {_format_num(f0 * scale_factor)} Hz)"
                )
            else:
                scale_label = (
                    f"Audible scaling ×{_format_num(scale_factor)} "
                    f"(off — direct {_format_num(f0)} Hz)"
                )
            st.checkbox(scale_label, key="audible_scaling")

            audio = _cached_synthesize_audio(
                waveform=waveform,
                f0=f0,
                fs=fs,
                amplitude=amplitude,
                phase_deg=phase_deg,
                audible_scaling=bool(st.session_state["audible_scaling"]),
                scale_factor=scale_factor,
            )

            orig_pill = (
                f"{audio['output_sample_rate'] / 1000:.1f} kHz · {audio['pcm_bit_depth']}-bit PCM · "
                f"Peak: {_format_num(audio['original_peak_hz'])} Hz · RMS: {audio['original_rms']:.3f}"
            )
            alias_pill = (
                f"{audio['output_sample_rate'] / 1000:.1f} kHz · {audio['pcm_bit_depth']}-bit PCM · "
                f"Peak: {_format_num(audio['aliased_peak_hz'])} Hz · RMS: {audio['aliased_rms']:.3f}"
            )

            with st.container(border=True):
                st.markdown(
                    '<div class="sav-audio-meta">'
                    '<div class="sav-audio-title">Original continuous signal</div>'
                    f'<div class="sav-audio-sub">{html.escape(audio["original_label"])}</div>'
                    f'<div class="sav-audio-pill">{html.escape(orig_pill)}</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.audio(audio["original_wav"], format="audio/wav")

            with st.container(border=True):
                st.markdown(
                    '<div class="sav-audio-meta">'
                    '<div class="sav-audio-title">Sampled / aliased signal</div>'
                    f'<div class="sav-audio-sub">{html.escape(audio["aliased_label"])}</div>'
                    f'<div class="sav-audio-pill">{html.escape(alias_pill)}</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.audio(audio["aliased_wav"], format="audio/wav")

    with col_interp:
        with st.container(border=True):
            st.markdown('<h2 class="sav-card-title">DSP interpretation</h2>', unsafe_allow_html=True)
            st.markdown(
                render_interpretation_panel_html(analysis, waveform=waveform),
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
