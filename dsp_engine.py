"""Core DSP engine for the Sampling and Aliasing Visualizer.

Provides continuous waveform synthesis, uniform discrete sampling,
Whittaker-Shannon ideal sinc reconstruction, aliasing frequency/phase analysis,
4-step educational interpretation, and WAV audio synthesis. Also includes
modular hooks for zero-order hold, linear interpolation, and FFT spectrum
computation for future extensions.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal


# ---------------------------------------------------------------------------
# UI Design System Tokens (aligned with Sampling & Aliasing Visualizer.html)
# ---------------------------------------------------------------------------

UI_COLORS: dict[str, str] = {
    "bg_app": "#F4F5F8",
    "bg_surface": "#FFFFFF",
    "border_subtle": "#D5D9E0",
    "border_control": "#C9CED8",
    "text_primary": "#14171F",
    "text_secondary": "#4A5160",
    "accent_primary": "#2457D6",
    "accent_soft": "#E8EEFC",
    "trace_continuous": "#2457D6",
    "trace_samples": "#0B6B5D",
    "trace_reconstructed": "#C2410C",
    "status_danger_bg": "#FDECEA",
    "status_danger_border": "#B42318",
    "status_danger_badge": "#B42318",
    "status_danger_text": "#7A1810",
    "status_warning_bg": "#FEF3C7",
    "status_warning_border": "#B45309",
    "status_warning_badge": "#B45309",
    "status_warning_text": "#78350F",
    "status_safe_bg": "#E8F5E9",
    "status_safe_border": "#1B7F37",
    "status_safe_badge": "#1B7F37",
    "status_safe_text": "#14532D",
}


# ---------------------------------------------------------------------------
# Educational Presets (aligned with Sampling & Aliasing Visualizer.html)
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    "oversampled": {
        "id": "oversampled",
        "title": "Oversampled safe regime",
        "subtitle": "f₀ 5 Hz · fs 40 Hz",
        "waveform": "Sine",
        "f0": 5.0,
        "fs": 40.0,
        "amplitude": 1.0,
        "phase_deg": 0.0,
        "duration": 0.5,
        "audible_scaling": True,
        "scale_factor": 50.0,
    },
    "nyquist_limit": {
        "id": "nyquist_limit",
        "title": "Critical Nyquist limit",
        "subtitle": "f₀ 10 Hz · fs 20 Hz",
        "waveform": "Sine",
        "f0": 10.0,
        "fs": 20.0,
        "amplitude": 1.0,
        "phase_deg": 0.0,
        "duration": 0.5,
        "audible_scaling": True,
        "scale_factor": 50.0,
    },
    "classic_undersampling": {
        "id": "classic_undersampling",
        "title": "Classic undersampling",
        "subtitle": "f₀ 10 Hz · fs 12 Hz → 2 Hz",
        "waveform": "Sine",
        "f0": 10.0,
        "fs": 12.0,
        "amplitude": 1.0,
        "phase_deg": 0.0,
        "duration": 0.5,
        "audible_scaling": True,
        "scale_factor": 50.0,
    },
    "square_wave_harmonics": {
        "id": "square_wave_harmonics",
        "title": "Square-wave harmonics",
        "subtitle": "f₀ 5 Hz · fs 18 Hz",
        "waveform": "Square",
        "f0": 5.0,
        "fs": 18.0,
        "amplitude": 1.0,
        "phase_deg": 0.0,
        "duration": 0.5,
        "audible_scaling": True,
        "scale_factor": 50.0,
    },
}


SUPPORTED_WAVEFORMS = ("Sine", "Square", "Triangle")


@dataclass(frozen=True)
class AliasResult:
    """Exact fundamental frequency folding solution."""

    k: int
    f_signed: float
    f_alias: float
    phase_inverted: bool
    is_dc_freeze: bool
    is_critical_nyquist: bool
    r_mod: float


def _canonical_waveform(waveform: str) -> str:
    """Normalize waveform names while validating supported choices."""
    key = waveform.strip().lower()
    if key == "sine":
        return "Sine"
    if key == "square":
        return "Square"
    if key in ("triangle", "triangular"):
        return "Triangle"
    raise ValueError(
        f"Unsupported waveform {waveform!r}. Expected one of: Sine, Square, Triangle."
    )


def _validate_params(
    f0: float, fs: float, amplitude: float, duration: float | None = None
) -> None:
    """Validate physical DSP parameter ranges."""
    if f0 <= 0.0:
        raise ValueError(f"Signal frequency f0 must be positive, got {f0}.")
    if fs <= 0.0:
        raise ValueError(f"Sampling frequency fs must be positive, got {fs}.")
    if amplitude < 0.0:
        raise ValueError(f"Amplitude must be non-negative, got {amplitude}.")
    if duration is not None and duration <= 0.0:
        raise ValueError(f"Duration must be positive, got {duration}.")


def _format_num(value: float, decimals: int = 2) -> str:
    """Format numbers cleanly without trailing zeroes."""
    rounded = round(float(value), decimals)
    if np.isclose(rounded, round(rounded), atol=1e-9):
        return f"{int(round(rounded))}"
    return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")


def generate_waveform(
    t: np.ndarray | float,
    waveform: str,
    f0: float,
    amplitude: float = 1.0,
    phase_deg: float = 0.0,
) -> np.ndarray:
    """Generate a continuous-time Sine, Square, or Triangle waveform.

    All waveforms follow the same phase convention so that at theta = 0,
    Sine and Triangle start at 0 with positive slope, and Square has rising
    edge at theta = 0.
    """
    if f0 <= 0.0:
        raise ValueError(f"Signal frequency f0 must be positive, got {f0}.")
    if amplitude < 0.0:
        raise ValueError(f"Amplitude must be non-negative, got {amplitude}.")

    canonical = _canonical_waveform(waveform)
    t_arr = np.asarray(t, dtype=float)
    theta = 2.0 * np.pi * float(f0) * t_arr + np.deg2rad(float(phase_deg))

    if canonical == "Sine":
        return float(amplitude) * np.sin(theta)
    if canonical == "Square":
        return float(amplitude) * signal.square(theta)
    # Shift symmetric triangle by +pi/2 so theta=0 starts at 0 with +slope.
    return float(amplitude) * signal.sawtooth(theta + np.pi / 2.0, width=0.5)


def compute_alias_properties(f0: float, fs: float) -> AliasResult:
    """Compute folding integer k, baseband alias frequency, and phase inversion."""
    _validate_params(f0, fs, amplitude=1.0)

    nyquist_rate = 2.0 * f0
    is_critical = bool(np.isclose(fs, nyquist_rate, rtol=0.0, atol=1e-9))

    if fs > nyquist_rate and not is_critical:
        return AliasResult(
            k=0,
            f_signed=float(f0),
            f_alias=float(f0),
            phase_inverted=False,
            is_dc_freeze=False,
            is_critical_nyquist=False,
            r_mod=float(f0),
        )

    if is_critical:
        return AliasResult(
            k=0,
            f_signed=float(f0),
            f_alias=float(f0),
            phase_inverted=False,
            is_dc_freeze=False,
            is_critical_nyquist=True,
            r_mod=float(f0),
        )

    k = int(np.floor(f0 / fs + 0.5))
    f_signed = float(f0 - k * fs)
    f_alias = float(abs(f_signed))
    is_dc_freeze = bool(np.isclose(f_alias, 0.0, rtol=0.0, atol=1e-9))
    if is_dc_freeze:
        f_alias = 0.0
        f_signed = 0.0

    is_half_nyquist = bool(np.isclose(f_alias, fs / 2.0, rtol=0.0, atol=1e-9))
    phase_inverted = bool(f_signed < 0.0 and not is_dc_freeze and not is_half_nyquist)

    if is_dc_freeze:
        r_mod = 0.0
    else:
        r_mod = float(f0 % fs)
        if np.isclose(r_mod, fs, rtol=0.0, atol=1e-9):
            r_mod = 0.0

    return AliasResult(
        k=k,
        f_signed=f_signed,
        f_alias=f_alias,
        phase_inverted=phase_inverted,
        is_dc_freeze=is_dc_freeze,
        is_critical_nyquist=False,
        r_mod=r_mod,
    )


def compute_fundamental_alias_signal(
    t: np.ndarray,
    f0: float,
    fs: float,
    amplitude: float = 1.0,
    phase_deg: float = 0.0,
) -> tuple[np.ndarray, AliasResult]:
    """Evaluate the theoretical baseband fundamental alias signal on time grid t."""
    alias = compute_alias_properties(f0, fs)
    t_arr = np.asarray(t, dtype=float)
    phi = np.deg2rad(float(phase_deg))

    if alias.is_dc_freeze:
        dc_val = float(amplitude) * float(np.sin(phi))
        return np.full_like(t_arr, dc_val, dtype=float), alias

    if alias.is_critical_nyquist:
        # At fs = 2*f0, samples (-1)^n * A*sin(phi) reconstruct to A*sin(phi)*cos(2*pi*f0*t)
        y = float(amplitude) * float(np.sin(phi)) * np.cos(2.0 * np.pi * f0 * t_arr)
        return y, alias

    if alias.phase_inverted:
        y = -float(amplitude) * np.sin(2.0 * np.pi * alias.f_alias * t_arr - phi)
        return y, alias

    y = float(amplitude) * np.sin(2.0 * np.pi * alias.f_alias * t_arr + phi)
    return y, alias


def simulate_sampling(
    waveform: str = "Sine",
    f0: float = 10.0,
    fs: float = 12.0,
    amplitude: float = 1.0,
    phase_deg: float = 0.0,
    duration: float = 0.5,
    num_cont_points: int = 2500,
    guard_samples: int = 24,
) -> dict[str, Any]:
    """Simulate continuous signal generation, uniform sampling, and sinc reconstruction.

    Uses extended guard samples outside [0, duration] so Whittaker-Shannon
    sinc interpolation accurately represents steady-state periodic reconstruction
    without artificial window-edge ringing.
    """
    _validate_params(f0, fs, amplitude, duration)
    canonical = _canonical_waveform(waveform)

    t_cont = np.linspace(0.0, float(duration), int(num_cont_points))
    x_cont = generate_waveform(t_cont, canonical, f0, amplitude, phase_deg)

    dt_sample = 1.0 / float(fs)
    t_samples = np.arange(0.0, float(duration) + 1e-12, dt_sample)
    # Clip minor floating-point overshoot beyond duration
    t_samples = t_samples[t_samples <= float(duration) + 1e-9]
    x_samples = generate_waveform(t_samples, canonical, f0, amplitude, phase_deg)

    # Extended sample grid for clean Whittaker-Shannon sinc interpolation
    n_visible = len(t_samples)
    n_ext = np.arange(-int(guard_samples), n_visible + int(guard_samples))
    t_ext = n_ext * dt_sample
    x_ext = generate_waveform(t_ext, canonical, f0, amplitude, phase_deg)

    sinc_kernel = np.sinc(float(fs) * (t_cont[:, None] - t_ext[None, :]))
    x_recon = sinc_kernel @ x_ext

    x_alias_fundamental, alias = compute_fundamental_alias_signal(
        t_cont, f0=f0, fs=fs, amplitude=amplitude, phase_deg=phase_deg
    )

    return {
        "waveform": canonical,
        "f0": float(f0),
        "fs": float(fs),
        "amplitude": float(amplitude),
        "phase_deg": float(phase_deg),
        "duration": float(duration),
        "t_cont": t_cont,
        "x_cont": x_cont,
        "t_samples": t_samples,
        "x_samples": x_samples,
        "x_recon": x_recon,
        "x_alias_fundamental": x_alias_fundamental,
        "alias": alias,
    }


def _compute_harmonic_aliases(
    waveform: str, f0: float, fs: float, max_order: int = 15
) -> list[dict[str, Any]]:
    """Compute folding behavior for odd harmonics (3f0, 5f0, ...) of Square/Triangle."""
    if waveform == "Sine":
        return []
    nyquist_freq = fs / 2.0
    folded: list[dict[str, Any]] = []
    for m in range(3, max_order + 1, 2):
        f_h = m * f0
        if f_h > nyquist_freq + 1e-9:
            k_m = int(np.floor(f_h / fs + 0.5))
            f_signed = f_h - k_m * fs
            f_fold = abs(f_signed)
            amp_weight = (1.0 / m) if waveform == "Square" else (1.0 / (m * m))
            folded.append(
                {
                    "order": m,
                    "harmonic_freq": float(f_h),
                    "k": k_m,
                    "alias_freq": float(f_fold),
                    "phase_inverted": bool(f_signed < 0.0 and not np.isclose(f_fold, 0.0)),
                    "relative_weight": float(amp_weight),
                }
            )
    return folded


def _build_interpretation_steps(
    waveform: str,
    f0: float,
    fs: float,
    num_samples: int,
    nyquist_rate: float,
    nyquist_freq: float,
    alias: AliasResult,
    regime_code: str,
    harmonic_aliases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the 4-step live DSP interpretation panel content."""
    wf_lower = waveform.lower()
    f0_s = _format_num(f0)
    fs_s = _format_num(fs)
    nyq_rate_s = _format_num(nyquist_rate)
    nyq_freq_s = _format_num(nyquist_freq)
    alias_s = _format_num(alias.f_alias)

    # Step 1: Nyquist check
    if regime_code == "undersampled":
        step1 = (
            f"fs = {fs_s} Hz is below 2f₀ = {nyq_rate_s} Hz, "
            f"so the {wf_lower} is undersampled."
        )
    elif regime_code == "critical_nyquist":
        step1 = (
            f"fs = {fs_s} Hz equals the exact Nyquist rate 2f₀ = {nyq_rate_s} Hz "
            f"(boundary of the Sampling Theorem)."
        )
    elif regime_code == "harmonic_aliasing":
        step1 = (
            f"fs = {fs_s} Hz exceeds 2f₀ = {nyq_rate_s} Hz for the fundamental, "
            f"but a {wf_lower} wave has infinite odd harmonics above fs/2 = {nyq_freq_s} Hz."
        )
    else:
        step1 = (
            f"fs = {fs_s} Hz exceeds 2f₀ = {nyq_rate_s} Hz "
            f"(Nyquist frequency fs/2 = {nyq_freq_s} Hz), satisfying the sampling theorem."
        )

    # Step 2: Fold to baseband
    if regime_code == "undersampled":
        step2 = (
            f"k = round({f0_s} / {fs_s}) = {alias.k}, "
            f"so f_alias = |{f0_s} − {alias.k} · {fs_s}| = {alias_s} Hz."
        )
    elif regime_code == "critical_nyquist":
        step2 = (
            f"Fundamental f₀ = {f0_s} Hz sits directly on the Nyquist folding limit "
            f"fs/2 = {nyq_freq_s} Hz (k = 0)."
        )
    elif harmonic_aliases:
        preview = ", ".join(
            f"{h['order']}f₀ ({_format_num(h['harmonic_freq'])} Hz → {_format_num(h['alias_freq'])} Hz)"
            for h in harmonic_aliases[:3]
        )
        step2 = f"Fundamental stays at {f0_s} Hz (k = 0), while higher harmonics fold: {preview}."
    else:
        step2 = (
            f"f₀ = {f0_s} Hz lies inside the first Nyquist zone [0, {nyq_freq_s} Hz], "
            f"so k = 0 and f_alias = {f0_s} Hz."
        )

    # Step 3: Phase test
    if alias.is_dc_freeze:
        step3 = (
            f"{f0_s} mod {fs_s} = 0: every sample lands at the exact same phase of the cycle "
            f"(0 Hz DC alias)."
        )
    elif alias.phase_inverted:
        r_mod_s = _format_num(alias.r_mod)
        step3 = (
            f"{f0_s} mod {fs_s} = {r_mod_s} exceeds fs/2 = {nyq_freq_s}, "
            f"so the alias appears inverted by 180°."
        )
    elif regime_code == "critical_nyquist":
        step3 = (
            "Sampling at 2 samples/cycle means amplitude depends on sin(φ): "
            "φ = 0° hits zero crossings, whereas φ = 90° captures peaks."
        )
    else:
        r_mod_s = _format_num(alias.r_mod)
        step3 = (
            f"{r_mod_s} ≤ fs/2 = {nyq_freq_s}: no fundamental phase inversion occurs."
        )

    # Step 4: Interpretation
    if alias.is_dc_freeze:
        step4 = (
            f"All {num_samples} samples share identical values, collapsing a {f0_s} Hz "
            f"oscillation into a constant 0 Hz (DC) line."
        )
    elif regime_code == "undersampled":
        inv_word = "inverted " if alias.phase_inverted else ""
        if waveform == "Sine":
            step4 = (
                f"The {num_samples} samples fit a {alias_s} Hz {inv_word}sine exactly. "
                f"From the samples alone, {f0_s} Hz cannot be told from {alias_s} Hz."
            )
        else:
            step4 = (
                f"The {num_samples} samples follow a {alias_s} Hz {inv_word}fundamental plus "
                f"folded odd harmonics, distorting the {wf_lower} shape."
            )
    elif regime_code == "critical_nyquist":
        step4 = (
            f"Right at fs = 2f₀, any phase offset φ = 0° causes complete signal loss in "
            f"uniform samples; strict oversampling (fs > 2f₀) is required in practice."
        )
    elif regime_code == "harmonic_aliasing":
        step4 = (
            f"The {f0_s} Hz pitch is preserved, but folded harmonics cause Gibbs-like ripple "
            f"and amplitude distortion in the reconstructed {wf_lower} waveform."
        )
    else:
        step4 = (
            f"Ideal Whittaker-Shannon sinc interpolation through the {num_samples} samples "
            f"recovers the {f0_s} Hz {wf_lower} waveform accurately."
        )

    return [
        {"number": 1, "title": "Nyquist check", "body": step1},
        {"number": 2, "title": "Fold to baseband", "body": step2},
        {"number": 3, "title": "Phase test", "body": step3},
        {"number": 4, "title": "Interpretation", "body": step4},
    ]


def analyze_sampling(sim: dict[str, Any]) -> dict[str, Any]:
    """Compute DSP metrics, status classification, KPI values, and 4-step narrative."""
    waveform: str = sim["waveform"]
    f0: float = sim["f0"]
    fs: float = sim["fs"]
    amplitude: float = sim["amplitude"]
    duration: float = sim["duration"]
    x_cont: np.ndarray = sim["x_cont"]
    x_recon: np.ndarray = sim["x_recon"]
    t_samples: np.ndarray = sim["t_samples"]
    alias: AliasResult = sim["alias"]

    nyquist_rate = 2.0 * f0
    nyquist_freq = fs / 2.0
    samples_per_cycle = fs / f0
    num_samples = int(len(t_samples))

    mse = float(np.mean((x_cont - x_recon) ** 2))
    signal_power = float(np.mean(x_cont ** 2))
    if signal_power <= 1e-15:
        snr_db = 0.0
    else:
        snr_db = float(10.0 * np.log10(signal_power / max(mse, 1e-15)))
        snr_db = float(np.clip(snr_db, -60.0, 120.0))

    harmonic_aliases = _compute_harmonic_aliases(waveform, f0, fs, max_order=15)
    low_order_harmonics_alias = any(h["order"] <= 9 for h in harmonic_aliases)

    f0_s = _format_num(f0)
    alias_s = _format_num(alias.f_alias)
    nyq_rate_s = _format_num(nyquist_rate)
    wf_lower = waveform.lower()

    if fs < nyquist_rate - 1e-9:
        regime_code = "undersampled"
        badge_text = "UNDERSAMPLED"
        banner_level = "danger"
        if alias.is_dc_freeze:
            banner_message = (
                f"fs < 2f₀: the {f0_s} Hz {wf_lower} aliases to 0 Hz (DC)."
            )
        elif alias.phase_inverted:
            banner_message = (
                f"fs < 2f₀: the {f0_s} Hz {wf_lower} aliases to {alias_s} Hz with phase inversion."
            )
        else:
            banner_message = (
                f"fs < 2f₀: the {f0_s} Hz {wf_lower} aliases down to {alias_s} Hz."
            )
    elif alias.is_critical_nyquist:
        regime_code = "critical_nyquist"
        badge_text = "CRITICAL NYQUIST"
        banner_level = "warning"
        banner_message = (
            f"fs = 2f₀: sampling right at the Nyquist limit ({nyq_rate_s} Hz); "
            f"phase alignment is critical."
        )
    elif waveform != "Sine" and low_order_harmonics_alias:
        regime_code = "harmonic_aliasing"
        badge_text = "HARMONIC ALIASING"
        banner_level = "warning"
        first_h = harmonic_aliases[0]
        banner_message = (
            f"fs > 2f₀ for fundamental, but odd harmonics fold back "
            f"(e.g., {first_h['order']}f₀ = {_format_num(first_h['harmonic_freq'])} Hz "
            f"→ {_format_num(first_h['alias_freq'])} Hz)."
        )
    elif waveform != "Sine":
        regime_code = "oversampled_fundamental"
        badge_text = "OVERSAMPLED (FUNDAMENTAL)"
        banner_level = "safe"
        banner_message = (
            f"fs > 2f₀: fundamental and low-order harmonics up to 9f₀ are resolved; "
            f"very high harmonics still fold weakly."
        )
    else:
        regime_code = "oversampled"
        badge_text = "OVERSAMPLED"
        banner_level = "safe"
        banner_message = (
            f"fs > 2f₀: Nyquist condition satisfied ({samples_per_cycle:.2f} samples/cycle) "
            f"— exact reconstruction."
        )

    steps = _build_interpretation_steps(
        waveform=waveform,
        f0=f0,
        fs=fs,
        num_samples=num_samples,
        nyquist_rate=nyquist_rate,
        nyquist_freq=nyquist_freq,
        alias=alias,
        regime_code=regime_code,
        harmonic_aliases=harmonic_aliases,
    )

    # Alias card subtitle matching HTML style ("k = 1, inverted" / "k = 0, exact")
    if alias.is_dc_freeze:
        alias_value_display = "0 Hz (DC)"
        alias_subtitle = f"k = {alias.k}, DC alias"
    elif alias.is_critical_nyquist:
        alias_value_display = f"{alias_s} Hz"
        alias_subtitle = "fs = 2f₀ boundary"
    elif alias.phase_inverted:
        alias_value_display = f"{alias_s} Hz"
        alias_subtitle = f"k = {alias.k}, inverted"
    elif regime_code == "undersampled":
        alias_value_display = f"{alias_s} Hz"
        alias_subtitle = f"k = {alias.k}, in-phase"
    else:
        alias_value_display = f"{alias_s} Hz"
        alias_subtitle = "k = 0, no folding"

    snr_rounded = round(snr_db, 1)
    if abs(snr_rounded) < 0.05:
        snr_str = "0.0 dB"
    elif snr_rounded < 0:
        snr_str = f"\u2212{abs(snr_rounded):.1f} dB"
    else:
        snr_str = f"{snr_rounded:.1f} dB"

    kpi_cards = [
        {
            "label": "Signal f₀",
            "value": f"{f0_s} Hz",
            "subtitle": f"{wf_lower}, A = {_format_num(amplitude)}",
        },
        {
            "label": "Sampling fs",
            "value": f"{_format_num(fs)} Hz",
            "subtitle": f"{num_samples} samples in {_format_num(duration)} s",
        },
        {
            "label": "Nyquist rate 2f₀",
            "value": f"{nyq_rate_s} Hz",
            "subtitle": f"fs/2 = {_format_num(nyquist_freq)} Hz",
        },
        {
            "label": "Apparent alias",
            "value": alias_value_display,
            "subtitle": alias_subtitle,
        },
        {
            "label": "Samples / cycle",
            "value": _format_num(samples_per_cycle, decimals=2),
            "subtitle": "fs / f₀",
        },
        {
            "label": "MSE · SNR",
            "value": f"{mse:.2f} · {snr_str}",
            "subtitle": "x(t) vs x̂(t)",
        },
    ]

    return {
        "nyquist_rate": float(nyquist_rate),
        "nyquist_freq": float(nyquist_freq),
        "samples_per_cycle": float(samples_per_cycle),
        "num_samples": num_samples,
        "mse": mse,
        "snr_db": snr_db,
        "regime_code": regime_code,
        "badge_text": badge_text,
        "banner_level": banner_level,
        "banner_message": banner_message,
        "harmonic_aliases": harmonic_aliases,
        "kpi_cards": kpi_cards,
        "steps": steps,
    }


def _pcm16_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Convert a floating-point signal in [-1, 1] into in-memory 16-bit PCM WAV bytes."""
    arr = np.asarray(samples, dtype=float)
    peak = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0

    if peak > 1e-6:
        # Apply 20 ms raised-cosine fade-in/fade-out to avoid playback clicks
        fade_len = min(int(0.02 * sample_rate), arr.size // 4)
        if fade_len > 0:
            ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_len) / fade_len))
            arr = arr.copy()
            arr[:fade_len] *= ramp
            arr[-fade_len:] *= ramp[::-1]
        norm = arr / peak * 0.85
    else:
        norm = np.zeros_like(arr)

    pcm = np.clip(np.round(norm * 32767.0), -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _measure_pcm_metrics(wav_bytes: bytes) -> tuple[float, float, int]:
    """Decode WAV bytes and measure normalized RMS, dominant FFT peak Hz, and frame count."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if n_frames == 0:
        return 0.0, 0.0, 0
    pcm_norm = np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0
    rms = float(np.sqrt(np.mean(pcm_norm ** 2)))
    if rms < 1e-6:
        return 0.0, 0.0, int(n_frames)
    freqs = np.fft.rfftfreq(pcm_norm.size, d=1.0 / float(sr))
    mag = np.abs(np.fft.rfft(pcm_norm))
    peak_hz = float(freqs[int(np.argmax(mag))])
    return rms, peak_hz, int(n_frames)


def _synthesize_harmonic_series_audio(
    t: np.ndarray,
    waveform: str,
    f0: float,
    fs: float,
    phase_deg: float,
    scale_factor: float,
    output_sample_rate: int,
    aliased: bool,
) -> np.ndarray:
    """Synthesize bandlimited audio for Sine, Square, or Triangle waves.

    When `aliased=True`, each harmonic below the audio DAC limit folds across fs
    so the listener hears genuine pitch folding and inharmonic harmonic aliasing.
    """
    phi = np.deg2rad(float(phase_deg))
    nyquist_audio = 0.45 * output_sample_rate

    if waveform == "Sine":
        alias = compute_alias_properties(f0, fs)
        if not aliased:
            freq = f0 * scale_factor
            return np.sin(2.0 * np.pi * freq * t + phi) if freq < nyquist_audio else np.zeros_like(t)

        if alias.is_dc_freeze:
            return np.zeros_like(t)
        if alias.is_critical_nyquist:
            gain = float(np.sin(phi))
            freq = f0 * scale_factor
            return gain * np.cos(2.0 * np.pi * freq * t) if freq < nyquist_audio else np.zeros_like(t)
        freq = alias.f_alias * scale_factor
        if freq >= nyquist_audio:
            return np.zeros_like(t)
        if alias.phase_inverted:
            return -np.sin(2.0 * np.pi * freq * t - phi)
        return np.sin(2.0 * np.pi * freq * t + phi)

    # Square or Triangle: odd harmonics m = 1, 3, ..., 15
    y = np.zeros_like(t, dtype=float)
    for idx, m in enumerate(range(1, 16, 2)):
        if waveform == "Square":
            coeff = 4.0 / (np.pi * m)
        else:
            sign = 1.0 if (idx % 2 == 0) else -1.0
            coeff = sign * (8.0 / (np.pi * np.pi * m * m))

        harm_f = m * f0
        if not aliased:
            audio_f = harm_f * scale_factor
            if 1.0 <= audio_f < nyquist_audio:
                y += coeff * np.sin(2.0 * np.pi * audio_f * t + m * phi)
            continue

        # Fold this harmonic around fs
        k_m = int(np.floor(harm_f / fs + 0.5))
        f_signed = harm_f - k_m * fs
        f_fold = abs(f_signed)
        audio_f = f_fold * scale_factor
        if audio_f < 1.0 or audio_f >= nyquist_audio:
            continue
        if np.isclose(f_fold, fs / 2.0, atol=1e-9):
            y += coeff * np.sin(m * phi) * np.cos(2.0 * np.pi * audio_f * t)
        elif f_signed < 0.0:
            y -= coeff * np.sin(2.0 * np.pi * audio_f * t - m * phi)
        else:
            y += coeff * np.sin(2.0 * np.pi * audio_f * t + m * phi)

    return y


def synthesize_audio_pair(
    waveform: str = "Sine",
    f0: float = 10.0,
    fs: float = 12.0,
    amplitude: float = 1.0,
    phase_deg: float = 0.0,
    audible_scaling: bool = True,
    scale_factor: float = 50.0,
    audio_duration: float = 1.5,
    output_sample_rate: int = 44100,
) -> dict[str, Any]:
    """Synthesize Original and Sampled/Aliased WAV audio streams with descriptive labels."""
    _validate_params(f0, fs, amplitude, audio_duration)
    canonical = _canonical_waveform(waveform)

    eff_scale = float(scale_factor) if audible_scaling else 1.0
    if eff_scale <= 0.0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}.")

    n_samples = int(round(float(audio_duration) * int(output_sample_rate)))
    t_audio = np.linspace(0.0, float(audio_duration), n_samples, endpoint=False)

    orig_sig = amplitude * _synthesize_harmonic_series_audio(
        t_audio,
        waveform=canonical,
        f0=f0,
        fs=fs,
        phase_deg=phase_deg,
        scale_factor=eff_scale,
        output_sample_rate=output_sample_rate,
        aliased=False,
    )
    alias_sig = amplitude * _synthesize_harmonic_series_audio(
        t_audio,
        waveform=canonical,
        f0=f0,
        fs=fs,
        phase_deg=phase_deg,
        scale_factor=eff_scale,
        output_sample_rate=output_sample_rate,
        aliased=True,
    )

    alias = compute_alias_properties(f0, fs)
    audio_f0 = f0 * eff_scale
    audio_f_alias = alias.f_alias * eff_scale

    original_label = f"{_format_num(audio_f0)} Hz tone"
    if alias.is_dc_freeze and canonical == "Sine":
        aliased_label = "0 Hz (DC) — silent / constant offset"
    elif alias.is_critical_nyquist and canonical == "Sine" and np.isclose(
        np.sin(np.deg2rad(phase_deg)), 0.0, atol=1e-6
    ):
        aliased_label = (
            f"{_format_num(audio_f_alias)} Hz Nyquist boundary — zero-crossing samples (silent)"
        )
    elif fs < 2.0 * f0 - 1e-9:
        if audio_f_alias < audio_f0 - 1e-6:
            aliased_label = f"{_format_num(audio_f_alias)} Hz tone: the pitch drops"
        else:
            aliased_label = f"{_format_num(audio_f_alias)} Hz folded alias tone"
    elif canonical != "Sine":
        aliased_label = f"{_format_num(audio_f0)} Hz tone with folded harmonic distortion"
    else:
        aliased_label = f"{_format_num(audio_f_alias)} Hz tone: faithful pitch reconstruction"

    orig_wav = _pcm16_wav_bytes(orig_sig, output_sample_rate)
    alias_wav = _pcm16_wav_bytes(alias_sig, output_sample_rate)
    orig_rms, orig_peak_hz, num_frames = _measure_pcm_metrics(orig_wav)
    alias_rms, alias_peak_hz, _ = _measure_pcm_metrics(alias_wav)

    return {
        "original_wav": orig_wav,
        "aliased_wav": alias_wav,
        "audio_f0": float(audio_f0),
        "audio_fs": float(fs * eff_scale),
        "audio_f_alias": float(audio_f_alias),
        "effective_scale": float(eff_scale),
        "original_label": original_label,
        "aliased_label": aliased_label,
        "output_sample_rate": int(output_sample_rate),
        "original_rms": orig_rms,
        "aliased_rms": alias_rms,
        "original_peak_hz": orig_peak_hz,
        "aliased_peak_hz": alias_peak_hz,
        "duration_sec": float(audio_duration),
        "num_frames": num_frames,
        "pcm_bit_depth": 16,
    }


# ---------------------------------------------------------------------------
# Modular Future Extension Hooks (Reconstruction Comparison & FFT Spectrum)
# ---------------------------------------------------------------------------


def reconstruct_zoh(
    t_cont: np.ndarray, t_samples: np.ndarray, x_samples: np.ndarray
) -> np.ndarray:
    """Zero-Order Hold (staircase) reconstruction hook for future extension."""
    if len(t_samples) == 0:
        return np.zeros_like(t_cont, dtype=float)
    idx = np.searchsorted(t_samples, t_cont, side="right") - 1
    idx = np.clip(idx, 0, len(x_samples) - 1)
    return np.asarray(x_samples[idx], dtype=float)


def reconstruct_linear(
    t_cont: np.ndarray, t_samples: np.ndarray, x_samples: np.ndarray
) -> np.ndarray:
    """First-Order Hold (linear interpolation) hook for future extension."""
    if len(t_samples) == 0:
        return np.zeros_like(t_cont, dtype=float)
    return np.interp(t_cont, t_samples, x_samples)


def compute_spectrum(
    x: np.ndarray, sample_rate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute single-sided FFT magnitude spectrum (freqs, linear_mag, db_mag)."""
    arr = np.asarray(x, dtype=float)
    n = arr.size
    if n == 0 or sample_rate <= 0.0:
        return np.array([]), np.array([]), np.array([])
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    spec = np.fft.rfft(arr)
    mag = np.abs(spec) / n
    if n > 1:
        mag[1:-1] *= 2.0
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    return freqs, mag, mag_db
