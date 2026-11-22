# Sampling and Aliasing Visualizer

Interactive Digital Signal Processing (CS-419 Open Ended Lab — Option 2) dashboard demonstrating the Nyquist–Shannon sampling theorem, aliasing, phase inversion, Whittaker–Shannon sinc reconstruction, harmonic folding, and audio pitch folding.

## Features
- **Signal Types**: Sine, Square, and Triangle waveforms with adjustable frequency $f_0$, sampling rate $f_s$, amplitude $A$, phase $\phi$, and time window $T$.
- **4 Educational Presets**: Oversampled safe regime, Critical Nyquist limit, Classic undersampling, and Square-wave harmonics.
- **Time-Domain Overlay**: Interactive Plotly visualization of continuous $x(t)$, discrete samples $x[n]$, Whittaker–Shannon reconstruction $\hat{x}(t)$, and fundamental alias curve.
- **Audio Comparison Studio**: Synthesizes playable 44.1 kHz 16-bit PCM WAV streams for Original vs. Sampled/Aliased signals with optional audible frequency scaling.
- **Live DSP Interpretation**: Dynamic Nyquist check, baseband folding math, phase test, MSE/SNR metrics, and harmonic aliasing breakdown.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run Tests
```bash
pytest -v
```
