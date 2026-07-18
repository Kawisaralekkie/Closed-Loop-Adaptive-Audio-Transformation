# รายละเอียดทางเทคนิคของระบบ (Technical Implementation Details)

เอกสารนี้สรุปรายละเอียดการ implement ที่สำคัญสำหรับการเขียนวิทยานิพนธ์

---

## 1. Butterworth Filter

| parameter | value |
|---|---|
| **Order** | 5 |
| **Implementation** | `scipy.signal.butter(N=5, ...)` + `sosfiltfilt` (zero-phase) |
| **Effective order** | 10 (forward-backward double-pass) |
| **Output format** | Second-Order Sections (SOS) |

### System Usage

| Tool | Filter Type | objectives |
|---|---|---|
| `MidBandAttenuationTool` | Lowpass | Cut High Frequency to reduce speech intelligibility |
| `MidBandAttenuationTool` | Bandpass | separate band for attenuation |
| `StrongBlurringTool` | Lowpass | Cut High Frequency (aggressive) |
| `StrongBlurringTool` | Highpass | separate high-band for mixing |
| `StrongBlurringTool` | Bandpass | mid-band boost, noise injection, band attenuation |

### Formulation

```
H(s) = 1 / (1 + (s/ωc)^(2N))   by N=5, ωc = cutoff frequency
```

Zero-phase filtering: `y = filtfilt(b, a, x)` → no phase distortion, -3dB at cutoff frequency

---

## 2. Source Separation (nussl)

| parameter | value |
|---|---|
| **Library** | nussl (Northwestern University Source Separation Library) |
| **Algorithm** | `nussl.separation.primitive.TimbreClustering` |
| **ประเภท** | Unsupervised (ไม่ต้อง pre-trained model) |
| **หลักการ** | Separating sounds based on timbre characteristics by use spectral clustering |
| **Timeout** | 30 seconds |
| **Min quality threshold** | 0.3 (ต่ำกว่านี้ fallback ไป StrongBlurringTool) |

### Flow

```
Input audio → nussl.AudioSignal → TimbreClustering → estimates[]
  ├─ estimates[0] = speech track     → blur → blurred_speech
  ├─ estimates[1] = residual track   → preserve (No edit)
  └─ remix = blurred_speech + residual
```

### Fallback conditions

1. nussl timeout (> 30s) → Use StrongBlurringTool instead
2. separation_quality_score < 0.3 → Use StrongBlurringTool instead
3. nussl import error → Use StrongBlurringTool instead

---

## 3. Noise Injection

| parameter | value|
|---|---|
| **ชนิด Noise** | Band-limited Gaussian (white) noise |
| **Generator** | `numpy.random.default_rng().standard_normal()` |
| **Distribution** | Gaussian (mean=0, std=1) |
| **Band filtering** | Butterworth bandpass order 5 |
| **Default band** | 700–2800 Hz |
| **SNR control** | adjust scale by following target SNR |

### สูตร Scale

```python
sig_power = mean(signal²)
target_noise_power = sig_power / 10^(snr_db / 10)
current_noise_power = mean(band_noise²)
scale = √(target_noise_power / current_noise_power)

output = signal + scale × band_noise
```

### Method

1. Create white Gaussian noise (Length equals signal)
2. Filter by Butterworth bandpass (700–2800 Hz, order 5)
3. Calculate scale factor จาก target SNR
4. Inject white noise to signal

---

## 4. WER/CER Normalization

| parameter | value |
|---|---|
| **Library** | `jiwer` (JiWER — Python Word Error Rate) |
| **ASR Model** | OpenAI Whisper `base` (74M parameters) |
| **Language** | English (`language="en"`) |
| **Precision** | FP32 (`fp16=False`) |
| **Output range** | [0, 1] (clamped) |

### Formulate WER (jiwer standard)

```
WER = (S + D + I) / N

S = substitutions (Replaced word)
D = deletions (Missing word)
I = insertions (Additional words)
N = จำนวนคำใน reference
```

### Normalization (clip to [0, 1])

```python
raw_wer = jiwer.wer(reference, hypothesis)
wer = np.clip(raw_wer, 0.0, 1.0)
```

**The reason for clip**: `jiwer` A value > 1.0 can be returned when the hypothesis is much longer than the reference (insertions > N)

### Edge Cases

| Situation | WER/CER | Reason |
|---|---|---|
| Both reference and hypothesis are empty | 0.0 | No speech  |
| Reference empty (original no speech) | 0.0 | bypass privacy |
| Hypothesis empty (blur all) | 1.0 | Maximum privacy |

### Flow

```
Original audio → Whisper base → reference text (lowercased, stripped)
Processed audio → Whisper base → hypothesis text (lowercased, stripped)
→ jiwer.wer(reference, hypothesis) → clip(0, 1) → WER
→ jiwer.cer(reference, hypothesis) → clip(0, 1) → CER
```

---

## 5. Psychoacoustic Features

| parameter | value |
|---|---|
| **External Library** | Not use (Manual calculate from numpy/scipy) |
| **Implementation** | Lightweight heuristic proxies |
| **Reason** | Require speech for in-loop GATE decision |

### 4 Features for calculation

#### 5.1 Short-term Loudness

```python
rms = √(mean(audio²))
short_term_loudness = 20 × log10(rms + 1e-12)  # dB scale
```

**Proxy for**: ISO 532-1 loudness (simplified)

#### 5.2 Sharpness Proxy

```python
spectrum = |FFT(audio)|
freqs = FFT frequencies
hf_mask = freqs ≥ 3000 Hz
sharpness_proxy = Σ(spectrum[hf_mask]²) / Σ(spectrum²)
```

**Proxy for**: DIN 45692 sharpness (ratio of HF energy)

#### 5.3 Roughness Proxy

```python
# Spectral flux between consecutive frames
frame_size = sr // 20  # 50ms frames
spectral_flux = std of frame-to-frame spectral differences
roughness_proxy = normalized spectral_flux
```

**Proxy for**: ECMA-418-2 roughness (spectral variation)

#### 5.4 Fluctuation Proxy

```python
frame_energies = [rms(frame_i) for each 50ms frame]
fluctuation_proxy = std(frame_energies) / (mean(frame_energies) + 1e-12)
```

**Proxy for**: Fluctuation strength (amplitude modulation depth)

### รวมเป็น S_psy

```python
loud_norm = clip(1 - |loudness + 20| / 60, 0, 1)
sharp_score = clip(1 - sharpness_proxy, 0, 1)
rough_score = clip(1 - roughness_proxy, 0, 1)
fluct_score = clip(1 - fluctuation_proxy, 0, 1)

S_psy = 0.25 × loud_norm + 0.25 × sharp_score + 0.25 × rough_score + 0.25 × fluct_score
```

### Limitation

- **No ISO/DIN standard implementation** — is lightweight proxy
- Not use auditory filterbank (gammatone/gammachirp)
- Not use time-varying loudness model
- Suite with fast GATE decision not suitable for perceptual quality reporting

---

## Summary of Dependencies

| Component | Library | Version | Noted |
|---|---|---|---|
| Butterworth filter | `scipy.signal` | ≥1.9 | `butter` + `sosfiltfilt` |
| Source separation | `nussl` | ≥1.1 | TimbreClustering (unsupervised) |
| Noise generation | `numpy` | ≥1.24 | `default_rng().standard_normal()` |
| WER/CER | `jiwer` | ≥3.0 | Standard WER/CER computation |
| ASR | `openai-whisper` | ≥20231117 | Model: `base` (74M params) |
| Psychoacoustic | `numpy` + `scipy` | — | Custom heuristic (no external lib) |
| Classification | `tensorflow-hub` | ≥0.14 | YAMNet v1 (521 classes) |
| Speaker embedding | `transformers` (HuggingFace) | ≥4.30 | WavLM-based speaker verification |
| VAD | `silero-vad` | v5.1 | Via `torch.hub` |
