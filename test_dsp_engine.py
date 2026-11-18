"""Unit tests for the Sampling and Aliasing Visualizer DSP engine."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from dsp_engine import (
    PRESETS,
    UI_COLORS,
    analyze_sampling,
    compute_alias_properties,
    compute_spectrum,
    generate_waveform,
    reconstruct_linear,
    reconstruct_zoh,
    simulate_sampling,
    synthesize_audio_pair,
)


def test_presets_execute_cleanly() -> None:
    """All 6 educational presets execute simulation, analysis, and audio synthesis."""
    assert len(PRESETS) == 6
    expected_ids = {
        "oversampled",
        "nyquist_limit",
        "classic_undersampling",
        "stroboscopic_freeze",
        "audible_pitch_folding",
        "square_wave_harmonics",
    }
    assert set(PRESETS.keys()) == expected_ids

    for preset_id, cfg in PRESETS.items():
        sim = simulate_sampling(
            waveform=cfg["waveform"],
            f0=cfg["f0"],
            fs=cfg["fs"],
            amplitude=cfg["amplitude"],
            phase_deg=cfg["phase_deg"],
            duration=cfg["duration"],
        )
        analysis = analyze_sampling(sim)
        audio = synthesize_audio_pair(
            waveform=cfg["waveform"],
            f0=cfg["f0"],
            fs=cfg["fs"],
            amplitude=cfg["amplitude"],
            phase_deg=cfg["phase_deg"],
            audible_scaling=cfg["audible_scaling"],
            scale_factor=cfg["scale_factor"],
            audio_duration=0.25,
        )

        assert sim["t_cont"].shape == sim["x_cont"].shape == sim["x_recon"].shape
        assert len(analysis["kpi_cards"]) == 6
        assert len(analysis["steps"]) == 4
        assert audio["original_wav"].startswith(b"RIFF"), preset_id
        assert audio["aliased_wav"].startswith(b"RIFF"), preset_id


def test_classic_undersampling_preset_exact_metrics() -> None:
    """Classic undersampling (f0=10 Hz, fs=12 Hz, Sine, T=0.5 s) matches HTML metrics."""
    cfg = PRESETS["classic_undersampling"]
    sim = simulate_sampling(
        waveform=cfg["waveform"],
        f0=cfg["f0"],
        fs=cfg["fs"],
        amplitude=cfg["amplitude"],
        phase_deg=cfg["phase_deg"],
        duration=cfg["duration"],
    )
    analysis = analyze_sampling(sim)
    alias = sim["alias"]

    assert analysis["num_samples"] == 7
    assert analysis["nyquist_rate"] == pytest.approx(20.0)
    assert analysis["nyquist_freq"] == pytest.approx(6.0)
    assert alias.f_alias == pytest.approx(2.0)
    assert alias.k == 1
    assert alias.phase_inverted is True
    assert analysis["samples_per_cycle"] == pytest.approx(1.2)
    assert analysis["mse"] == pytest.approx(1.0, abs=0.03)
    assert analysis["snr_db"] == pytest.approx(-3.0, abs=0.2)
    assert analysis["regime_code"] == "undersampled"
    assert analysis["badge_text"] == "UNDERSAMPLED"
    assert analysis["banner_level"] == "danger"
    assert np.allclose(sim["x_recon"], sim["x_alias_fundamental"], atol=0.05)


def test_oversampled_preset_accurate_reconstruction() -> None:
    """Oversampled regime (f0=5 Hz, fs=40 Hz, Sine) reconstructs accurately."""
    cfg = PRESETS["oversampled"]
    sim = simulate_sampling(
        waveform=cfg["waveform"],
        f0=cfg["f0"],
        fs=cfg["fs"],
        amplitude=cfg["amplitude"],
        phase_deg=cfg["phase_deg"],
        duration=cfg["duration"],
    )
    analysis = analyze_sampling(sim)
    alias = sim["alias"]

    assert analysis["regime_code"] == "oversampled"
    assert analysis["badge_text"] == "OVERSAMPLED"
    assert analysis["banner_level"] == "safe"
    assert alias.f_alias == pytest.approx(5.0)
    assert alias.phase_inverted is False
    assert analysis["mse"] < 1e-3
    assert analysis["snr_db"] > 30.0


def test_stroboscopic_freeze_preset() -> None:
    """Sampling at f0=fs=10 Hz produces 0 Hz DC freeze with zero samples at phi=0."""
    cfg = PRESETS["stroboscopic_freeze"]
    sim = simulate_sampling(
        waveform=cfg["waveform"],
        f0=cfg["f0"],
        fs=cfg["fs"],
        amplitude=cfg["amplitude"],
        phase_deg=cfg["phase_deg"],
        duration=cfg["duration"],
    )
    analysis = analyze_sampling(sim)
    alias = sim["alias"]

    assert alias.f_alias == pytest.approx(0.0)
    assert alias.is_dc_freeze is True
    assert alias.phase_inverted is False
    assert np.allclose(sim["x_samples"], 0.0, atol=1e-9)
    assert np.allclose(sim["x_alias_fundamental"], 0.0, atol=1e-9)
    assert analysis["regime_code"] == "undersampled"


def test_critical_nyquist_limit_preset() -> None:
    """Critical Nyquist (fs = 2*f0) detects boundary regime and phase sensitivity."""
    cfg = PRESETS["nyquist_limit"]
    sim_zero = simulate_sampling(
        waveform=cfg["waveform"],
        f0=cfg["f0"],
        fs=cfg["fs"],
        amplitude=cfg["amplitude"],
        phase_deg=0.0,
        duration=cfg["duration"],
    )
    analysis_zero = analyze_sampling(sim_zero)

    assert analysis_zero["regime_code"] == "critical_nyquist"
    assert analysis_zero["badge_text"] == "CRITICAL NYQUIST"
    assert analysis_zero["banner_level"] == "warning"
    assert np.allclose(sim_zero["x_samples"], 0.0, atol=1e-9)

    sim_90 = simulate_sampling(
        waveform=cfg["waveform"],
        f0=cfg["f0"],
        fs=cfg["fs"],
        amplitude=cfg["amplitude"],
        phase_deg=90.0,
        duration=cfg["duration"],
    )
    assert np.allclose(np.abs(sim_90["x_samples"]), 1.0, atol=1e-9)
    # Sinc series at exact Nyquist (fs = 2*f0) converges conditionally with O(1/G) edge bias
    assert np.mean((sim_90["x_recon"] - sim_90["x_cont"]) ** 2) < 0.005
    assert np.allclose(sim_90["x_recon"], sim_90["x_cont"], atol=0.12)


def test_waveforms_amplitude_phase_and_harmonic_aliasing() -> None:
    """Verify Sine, Square, and Triangle bounds, phases, and harmonic aliasing."""
    t = np.linspace(0.0, 0.2, 2000, endpoint=False)
    for wf in ("Sine", "Square", "Triangle", "Triangular"):
        y = generate_waveform(t, wf, f0=10.0, amplitude=1.75, phase_deg=0.0)
        assert np.max(np.abs(y)) <= 1.75 + 1e-9
        assert np.max(np.abs(y)) >= 1.70

    # Sine and Triangle start at 0 at t=0, phi=0 and rise positively
    t_q = np.array([0.0, 0.025])  # 0 and quarter period of 10 Hz
    tri_q = generate_waveform(t_q, "Triangular", f0=10.0, amplitude=2.0, phase_deg=0.0)
    assert tri_q[0] == pytest.approx(0.0, abs=1e-9)
    assert tri_q[1] == pytest.approx(2.0, abs=1e-9)

    # Phase shift of 90 deg puts Sine and Triangle at +A at t=0
    assert generate_waveform(np.array([0.0]), "Sine", f0=10.0, amplitude=1.5, phase_deg=90.0)[0] == pytest.approx(1.5)
    assert generate_waveform(np.array([0.0]), "Triangle", f0=10.0, amplitude=1.5, phase_deg=90.0)[0] == pytest.approx(1.5)

    # Square and Triangle detect harmonic_aliasing when fs > 2*f0 but low harmonics fold
    for wf in ("Square", "Triangle"):
        sim = simulate_sampling(waveform=wf, f0=5.0, fs=18.0, duration=0.5)
        analysis = analyze_sampling(sim)
        assert analysis["regime_code"] == "harmonic_aliasing"
        assert analysis["badge_text"] == "HARMONIC ALIASING"
        assert analysis["banner_level"] == "warning"
        assert len(analysis["harmonic_aliases"]) > 0
        # 3*5 = 15 Hz folds across fs=18 Hz to 3 Hz
        assert analysis["harmonic_aliases"][0]["order"] == 3
        assert analysis["harmonic_aliases"][0]["alias_freq"] == pytest.approx(3.0)


def test_audio_synthesis_wav_format_and_labels() -> None:
    """Synthesized WAV bytes have valid headers, 44.1 kHz rate, and expected labels."""
    audio = synthesize_audio_pair(
        waveform="Sine",
        f0=10.0,
        fs=12.0,
        amplitude=1.0,
        phase_deg=0.0,
        audible_scaling=True,
        scale_factor=50.0,
        audio_duration=0.5,
        output_sample_rate=44100,
    )

    assert audio["original_label"] == "500 Hz tone"
    assert audio["aliased_label"] == "100 Hz tone: the pitch drops"
    assert audio["audio_f0"] == pytest.approx(500.0)
    assert audio["audio_f_alias"] == pytest.approx(100.0)

    for key in ("original_wav", "aliased_wav"):
        wav_bytes = audio[key]
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 44100
            assert wf.getnframes() == 22050
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
            assert np.max(np.abs(pcm)) > 1000

    # Unscaled audible pitch folding preset
    cfg = PRESETS["audible_pitch_folding"]
    direct_audio = synthesize_audio_pair(
        waveform=cfg["waveform"],
        f0=cfg["f0"],
        fs=cfg["fs"],
        audible_scaling=cfg["audible_scaling"],
        scale_factor=cfg["scale_factor"],
        audio_duration=0.25,
    )
    assert direct_audio["original_label"] == "880 Hz tone"
    assert direct_audio["aliased_label"] == "120 Hz tone: the pitch drops"


def test_alias_floating_point_stability_and_validation() -> None:
    """Ensure float modulo edge cases (e.g. 0.3 vs 0.1) and validation work properly."""
    dc_alias = compute_alias_properties(f0=0.3, fs=0.1)
    assert dc_alias.is_dc_freeze is True
    assert dc_alias.f_alias == pytest.approx(0.0)
    assert dc_alias.phase_inverted is False

    with pytest.raises(ValueError):
        simulate_sampling(f0=-5.0, fs=10.0)
    with pytest.raises(ValueError):
        simulate_sampling(f0=5.0, fs=0.0)
    with pytest.raises(ValueError):
        generate_waveform(np.array([0.0]), "Sawtooth", f0=5.0)


def test_future_extension_hooks_and_design_tokens() -> None:
    """Verify modular ZOH, linear interpolation, spectrum hooks, and UI color tokens."""
    assert UI_COLORS["trace_continuous"] == "#2457D6"
    assert UI_COLORS["trace_samples"] == "#0B6B5D"
    assert UI_COLORS["trace_reconstructed"] == "#C2410C"

    sim = simulate_sampling(waveform="Sine", f0=5.0, fs=40.0, duration=0.5)
    zoh = reconstruct_zoh(sim["t_cont"], sim["t_samples"], sim["x_samples"])
    lin = reconstruct_linear(sim["t_cont"], sim["t_samples"], sim["x_samples"])
    assert zoh.shape == sim["t_cont"].shape
    assert lin.shape == sim["t_cont"].shape

    freqs, mag, mag_db = compute_spectrum(sim["x_cont"], sample_rate=5000.0)
    assert freqs.size == mag.size == mag_db.size
    peak_freq = freqs[int(np.argmax(mag))]
    assert peak_freq == pytest.approx(6.0, abs=3.0)


def _decode_wav_fft(wav_bytes: bytes) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Decode WAV bytes and return (sample_rate, pcm_float, fft_freqs, fft_mag)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        pcm = np.frombuffer(wf.readframes(n_frames), dtype="<i2").astype(float) / 32768.0
    freqs = np.fft.rfftfreq(pcm.size, d=1.0 / sr)
    mag = np.abs(np.fft.rfft(pcm)) / max(pcm.size, 1)
    return sr, pcm, freqs, mag


def test_signals_and_audio_before_and_after_across_all_presets() -> None:
    """Verify time-domain signal fidelity and before/after WAV FFT spectra across presets."""
    # 1. Time-domain sample & sinc-interpolation consistency across all presets & waveforms
    for preset_id, cfg in PRESETS.items():
        sim = simulate_sampling(
            waveform=cfg["waveform"],
            f0=cfg["f0"],
            fs=cfg["fs"],
            amplitude=cfg["amplitude"],
            phase_deg=cfg["phase_deg"],
            duration=cfg["duration"],
        )
        t_c = sim["t_cont"]
        t_s = sim["t_samples"]
        # Reconstructed sinc curve must pass through discrete sample values x[n]
        x_recon_at_samples = np.interp(t_s, t_c, sim["x_recon"])
        assert np.allclose(x_recon_at_samples, sim["x_samples"], atol=1e-2), preset_id
        if cfg["waveform"] in ("Sine", "Triangle"):
            x_cont_at_samples = np.interp(t_s, t_c, sim["x_cont"])
            assert np.allclose(x_cont_at_samples, sim["x_samples"], atol=1e-2), preset_id

    for wf in ("Sine", "Square", "Triangle"):
        sim_wf = simulate_sampling(waveform=wf, f0=6.0, fs=30.0, duration=0.5)
        x_recon_at_s = np.interp(sim_wf["t_samples"], sim_wf["t_cont"], sim_wf["x_recon"])
        assert np.allclose(x_recon_at_s, sim_wf["x_samples"], atol=2e-2), wf

    # 2. Audio verification: Classic undersampling (500 Hz -> 100 Hz inverted)
    cu = PRESETS["classic_undersampling"]
    cu_audio = synthesize_audio_pair(
        waveform=cu["waveform"],
        f0=cu["f0"],
        fs=cu["fs"],
        amplitude=cu["amplitude"],
        phase_deg=cu["phase_deg"],
        audible_scaling=cu["audible_scaling"],
        scale_factor=cu["scale_factor"],
    )
    assert cu_audio["duration_sec"] == pytest.approx(1.5)
    assert cu_audio["num_frames"] == 66150
    assert cu_audio["pcm_bit_depth"] == 16
    assert cu_audio["original_peak_hz"] == pytest.approx(500.0, abs=1.0)
    assert cu_audio["aliased_peak_hz"] == pytest.approx(100.0, abs=1.0)
    assert cu_audio["original_rms"] > 0.4
    assert cu_audio["aliased_rms"] > 0.4

    # Check phase inversion of decoded 100 Hz aliased audio (-sin(2*pi*100*t))
    sr, pcm_alias, _, _ = _decode_wav_fft(cu_audio["aliased_wav"])
    t_mid = np.arange(pcm_alias.size) / sr
    # Exclude 20 ms fade edges
    sl = slice(int(0.05 * sr), int(1.45 * sr))
    ref_sin = np.sin(2.0 * np.pi * 100.0 * t_mid[sl])
    corr = float(np.dot(pcm_alias[sl], ref_sin))
    assert corr < -100.0

    # 3. Oversampled safe regime (250 Hz -> 250 Hz)
    ov = PRESETS["oversampled"]
    ov_audio = synthesize_audio_pair(
        waveform=ov["waveform"],
        f0=ov["f0"],
        fs=ov["fs"],
        audible_scaling=ov["audible_scaling"],
        scale_factor=ov["scale_factor"],
    )
    assert ov_audio["original_peak_hz"] == pytest.approx(250.0, abs=1.0)
    assert ov_audio["aliased_peak_hz"] == pytest.approx(250.0, abs=1.0)

    # 4. Audible pitch folding (880 Hz -> 120 Hz)
    ap = PRESETS["audible_pitch_folding"]
    ap_audio = synthesize_audio_pair(
        waveform=ap["waveform"],
        f0=ap["f0"],
        fs=ap["fs"],
        audible_scaling=ap["audible_scaling"],
        scale_factor=ap["scale_factor"],
    )
    assert ap_audio["original_peak_hz"] == pytest.approx(880.0, abs=1.0)
    assert ap_audio["aliased_peak_hz"] == pytest.approx(120.0, abs=1.0)

    # 5. Stroboscopic freeze & Critical Nyquist (phi=0): Original 500 Hz -> Aliased silent
    for silent_id in ("stroboscopic_freeze", "nyquist_limit"):
        sc = PRESETS[silent_id]
        s_audio = synthesize_audio_pair(
            waveform=sc["waveform"],
            f0=sc["f0"],
            fs=sc["fs"],
            phase_deg=sc["phase_deg"],
            audible_scaling=sc["audible_scaling"],
            scale_factor=sc["scale_factor"],
        )
        assert s_audio["original_peak_hz"] == pytest.approx(500.0, abs=1.0)
        assert s_audio["aliased_rms"] == pytest.approx(0.0, abs=1e-9)
        assert s_audio["aliased_peak_hz"] == pytest.approx(0.0, abs=1e-9)

    # 6. Square-wave harmonics: verify odd harmonics before and folded harmonics after
    sq = PRESETS["square_wave_harmonics"]
    sq_audio = synthesize_audio_pair(
        waveform=sq["waveform"],
        f0=sq["f0"],
        fs=sq["fs"],
        audible_scaling=sq["audible_scaling"],
        scale_factor=sq["scale_factor"],
    )
    _, _, freqs_o, mag_o = _decode_wav_fft(sq_audio["original_wav"])
    _, _, freqs_a, mag_a = _decode_wav_fft(sq_audio["aliased_wav"])

    def _bin_amp(freqs_arr: np.ndarray, mag_arr: np.ndarray, target_hz: float) -> float:
        idx = int(np.argmin(np.abs(freqs_arr - target_hz)))
        return float(mag_arr[idx])

    # Original square wave (5 Hz * 50 = 250 Hz): peaks at 250, 750, 1250 Hz
    assert _bin_amp(freqs_o, mag_o, 250.0) > 0.25
    assert _bin_amp(freqs_o, mag_o, 750.0) > 0.08
    assert _bin_amp(freqs_o, mag_o, 1250.0) > 0.04

    # Aliased square wave (fs*50 = 900 Hz): 750 Hz folds to 150 Hz, 1250 Hz folds to 350 Hz
    assert _bin_amp(freqs_a, mag_a, 250.0) > 0.20
    assert _bin_amp(freqs_a, mag_a, 150.0) > 0.06
    assert _bin_amp(freqs_a, mag_a, 350.0) > 0.03

