"""UI and integration tests for the Sampling and Aliasing Visualizer Streamlit app."""

from __future__ import annotations

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

import app
from dsp_engine import PRESETS, UI_COLORS, analyze_sampling, simulate_sampling


APP_PATH = str(Path(__file__).resolve().parent / "app.py")


def test_build_time_domain_figure_toggles_and_presets() -> None:
    """Plotly figure builder respects trace visibility toggles and design colors."""
    sim = simulate_sampling(waveform="Sine", f0=10.0, fs=12.0, duration=0.5)

    # All 4 layers enabled: continuous, recon, alias, stems + markers = 5 traces
    fig_all = app.build_time_domain_figure(
        sim,
        show_continuous=True,
        show_samples=True,
        show_recon=True,
        show_alias_overlay=True,
    )
    assert len(fig_all.data) == 5
    assert fig_all.data[0].line.color == UI_COLORS["trace_continuous"]
    assert fig_all.data[1].line.color == UI_COLORS["trace_reconstructed"]
    assert fig_all.data[4].marker.color == UI_COLORS["trace_samples"]

    # None enabled: 0 traces
    fig_none = app.build_time_domain_figure(
        sim,
        show_continuous=False,
        show_samples=False,
        show_recon=False,
        show_alias_overlay=False,
    )
    assert len(fig_none.data) == 0

    # Alias overlay alone promotes color to trace_reconstructed
    fig_alias_only = app.build_time_domain_figure(
        sim,
        show_continuous=False,
        show_samples=False,
        show_recon=False,
        show_alias_overlay=True,
    )
    assert len(fig_alias_only.data) == 1
    assert fig_alias_only.data[0].line.color == UI_COLORS["trace_reconstructed"]


def test_html_builders_content_and_escaping() -> None:
    """HTML builders emit expected regime badges, KPI cards, steps, and harmonic pills."""
    header_html = app.render_header_html()
    assert "Sampling and Aliasing Visualizer" in header_html

    # Classic undersampling banner & KPI grid
    sim = simulate_sampling(waveform="Sine", f0=10.0, fs=12.0, duration=0.5)
    analysis = analyze_sampling(sim)
    banner_html = app.render_status_banner_html(analysis)
    assert "UNDERSAMPLED" in banner_html
    assert UI_COLORS["status_danger_bg"] in banner_html

    kpi_html = app.render_kpi_grid_html(analysis["kpi_cards"])
    assert "10 Hz" in kpi_html
    assert "12 Hz" in kpi_html
    assert "20 Hz" in kpi_html
    assert "2 Hz" in kpi_html

    legend_html = app.render_time_domain_header_html()
    assert "Time-domain view" in legend_html
    assert UI_COLORS["trace_continuous"] in legend_html

    # Escaping test on synthetic message
    fake_analysis = dict(analysis)
    fake_analysis["banner_message"] = "<script>alert(1)</script>"
    escaped_banner = app.render_status_banner_html(fake_analysis)
    assert "<script>" not in escaped_banner
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in escaped_banner

    # Square wave harmonic pills
    sim_sq = simulate_sampling(waveform="Square", f0=5.0, fs=18.0, duration=0.5)
    analysis_sq = analyze_sampling(sim_sq)
    interp_sq = app.render_interpretation_panel_html(analysis_sq, waveform="Square")
    assert "Folded odd harmonics" in interp_sq
    assert "15 Hz &rarr; 3 Hz" in interp_sq


def test_streamlit_apptest_presets_waveforms_and_sync(monkeypatch) -> None:
    """End-to-end Streamlit AppTest runs presets, waveforms, toggles, and numeric sync."""
    monkeypatch.chdir(os.path.dirname(APP_PATH))
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()

    assert not at.exception
    assert at.session_state["active_preset"] == "classic_undersampling"
    assert at.session_state["waveform"] == "Sine"
    assert at.session_state["f0"] == 10.0
    assert at.session_state["fs"] == 12.0

    # Click through every educational preset (including high-freq 880/1000 and Square)
    for preset_id, cfg in PRESETS.items():
        at.sidebar.button(key=f"preset_{preset_id}").click().run()
        assert not at.exception, f"Exception on preset {preset_id}: {at.exception}"
        assert at.session_state["active_preset"] == preset_id
        assert at.session_state["waveform"] == cfg["waveform"]
        assert at.session_state["f0"] == float(cfg["f0"])
        assert at.session_state["fs"] == float(cfg["fs"])
        assert at.session_state["duration"] == float(cfg["duration"])

    # Switch back to oversampled then test waveform buttons
    at.sidebar.button(key="preset_oversampled").click().run()
    assert not at.exception
    for wf in ("Square", "Triangle", "Sine"):
        at.sidebar.button(key=f"wf_{wf.lower()}").click().run()
        assert not at.exception
        assert at.session_state["waveform"] == wf
        assert at.session_state["active_preset"] == "custom"

    # Test synchronized number_input -> slider & dynamic high-frequency expansion
    at.sidebar.number_input(key="f0_num").set_value(500.0).run()
    assert not at.exception
    assert at.session_state["f0"] == 500.0
    assert at.session_state["f0_slider"] == 500.0

    # Return via slider -> number_input
    at.sidebar.slider(key="f0_slider").set_value(25.0).run()
    assert not at.exception
    assert at.session_state["f0"] == 25.0
    assert at.session_state["f0_num"] == 25.0

    # Toggle visibility checkboxes and audible scaling
    for cb_key in (
        "show_continuous",
        "show_samples",
        "show_recon",
        "show_alias_overlay",
    ):
        at.sidebar.checkbox(key=cb_key).uncheck().run()
        assert not at.exception
        assert at.session_state[cb_key] is False
        at.sidebar.checkbox(key=cb_key).check().run()
        assert not at.exception
        assert at.session_state[cb_key] is True

    at.checkbox(key="audible_scaling").uncheck().run()
    assert not at.exception
    assert at.session_state["audible_scaling"] is False
    at.checkbox(key="audible_scaling").check().run()
    assert not at.exception
    assert at.session_state["audible_scaling"] is True
