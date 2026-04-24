# Isolated Word Recognition using Vector Quantization

Speech recognition system using **Vector Quantization (VQ)** and **LPC features**. Records 10 words × 15 times, trains codebooks with the LBG algorithm, and evaluates recognition accuracy via confusion matrices.

Built in Python, designed to integrate as a ROS 2 publisher node for the Puzzlebot navigation challenge.

---

## Requirements

- Python 3.10+
- A microphone connected to your computer
- `libportaudio2` system library (for audio recording)

```bash
sudo apt install libportaudio2
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Step 1 — record your voice (15 samples × 10 words)
python main.py record

# Step 2 — train VQ codebooks (sizes 16, 32, 64)
python main.py train

# Step 3 — evaluate and print confusion matrices
python main.py evaluate
```

Or run steps 2 and 3 together after recording:

```bash
python main.py all
```

---

## Vocabulary

The 10 words used for the Puzzlebot navigation challenge:

| # | Word | # | Word |
|---|------|---|------|
| 1 | start | 6 | right |
| 2 | stop | 7 | forward |
| 3 | lift | 8 | back |
| 4 | drop | 9 | faster |
| 5 | left | 10 | slower |

To change the words, edit `WORDS` in `config.py`.

---

## Recording

```bash
python main.py record
```

The script walks you through each word interactively:

1. Shows the word in uppercase and waits for **Enter**.
2. Asks you to press **Enter** again before each of the 15 recordings.
3. Each recording lasts **2 seconds** — say the word once, clearly.
4. Files are saved automatically to `recordings/<word>/01.wav … 15.wav`.

**Tips for clean recordings:**
- Use a quiet room with no background noise.
- Same speaker for all 150 recordings.
- Say the word once, naturally, in the middle of the 2-second window.
- Keep the same distance from the microphone throughout.
- If a take goes wrong, the script lets you re-record it.

**Resulting file layout:**

```
recordings/
├── start/
│   ├── 01.wav  ← training (files 01–10)
│   ├── ...
│   └── 15.wav  ← testing  (files 11–15)
├── stop/
│   └── ...
└── ...
```

---

## Training

```bash
python main.py train
```

Uses the first **10 recordings** per word to train three codebooks (sizes 16, 32, 64).

**What happens internally:**

1. Load `.wav` at 16 kHz.
2. Apply pre-emphasis filter `H(z) = 1 − 0.95z⁻¹`.
3. Detect the voiced segment using frame energy (20 dB above noise floor).
4. Split into **320-sample Hamming-windowed frames**, hop of **128 samples**.
5. Compute **LPC order-12** coefficients per frame (autocorrelation + Levinson-Durbin).
6. Convert LPC → **Line Spectral Frequencies (LSF)**.
7. Pool all LSF vectors from all 10 training files → run **LBG** clustering.
8. For each codebook entry, store the mean LPC coefficients and gain of its cluster (used for Itakura-Saito distance during recognition).

**Output:**

```
codebooks/
├── start_16.npz   start_32.npz   start_64.npz
├── stop_16.npz    stop_32.npz    stop_64.npz
└── ...
```

---

## Evaluation

```bash
python main.py evaluate          # all three sizes
python main.py evaluate 32       # size 32 only
python main.py evaluate 16 64    # sizes 16 and 64
```

Uses the last **5 recordings** (files `11.wav – 15.wav`) per word.

**Classification process per utterance:**

1. Extract features (same pipeline as training).
2. For every frame, compute the **Itakura-Saito (IS) spectral distortion** against every codeword in each word's codebook.
3. Take the minimum IS distance per frame (nearest codeword).
4. Average over all frames → per-word distortion score.
5. Predict the word with the **lowest average IS distortion**.
6. Accumulate results into a **10×10 confusion matrix**.

**Console output example:**

```
=== Codebook size 32 | Accuracy: 84.0% ===
Confusion matrix (rows=true, cols=predicted):
          start     stop     lift  ...
    start     5        0        0  ...
     stop     0        4        1  ...
```

**Plots saved to:**

```
results/
├── confusion_16.png
├── confusion_32.png
└── confusion_64.png
```

---

## Generating the PDF Report

```bash
python generate_pdf.py "Your Name" "A00000000"
```

Generates `Practica1_VQ_Reconocimiento.pdf`. If confusion matrix images exist in `results/`, they are embedded automatically.

---

## File Reference

| File | Purpose |
|---|---|
| `config.py` | All constants: sample rate, frame size, hop, LPC order, codebook sizes, paths |
| `features.py` | Signal processing: pre-emphasis, framing, VAD, LPC, LSF, IS distance |
| `vq.py` | LBG vector quantization algorithm |
| `train.py` | Feature extraction + LBG training → saves `.npz` codebooks |
| `recognize.py` | IS-distance classification + confusion matrix plotting |
| `record_words.py` | Interactive CLI for recording all 150 utterances |
| `main.py` | Unified entry point |
| `generate_pdf.py` | Generates the Spanish PDF report |

---

## Configuration

| Parameter | Value | Description |
|---|---|---|
| `SAMPLE_RATE` | 16 000 Hz | Recording and analysis sample rate |
| `FRAME_SIZE` | 320 samples | ~20 ms Hamming window |
| `HOP_SIZE` | 128 samples | ~8 ms frame shift |
| `LPC_ORDER` | 12 | LPC analysis order |
| `PRE_EMPHASIS_COEF` | 0.95 | Pre-emphasis filter coefficient |
| `CODEBOOK_SIZES` | [16, 32, 64] | VQ codebook sizes to train and compare |
| `N_TRAIN` | 10 | Training recordings per word |
| `N_TEST` | 5 | Test recordings per word |

---

## Signal Processing Pipeline

```
Raw audio (16 kHz)
       │
       ▼
Pre-emphasis:  H(z) = 1 − 0.95·z⁻¹
       │
       ▼
Voice Activity Detection (energy in dB, 20 dB threshold above noise floor)
       │
       ▼
Framing: 320-sample Hamming windows, 128-sample hop
       │
       ▼
LPC analysis (order 12): autocorrelation + Levinson-Durbin recursion
       │
       ▼
LSF conversion: companion polynomial root-finding
       │  (Euclidean distance used for LBG clustering)
       ▼
LBG → codebook of size 16 / 32 / 64
       │  (mean LPC per cluster stored for recognition)
       ▼
Recognition: Itakura-Saito spectral distortion (via FFT)
       │
       ▼
10×10 Confusion matrix
```

---

## Notes on Codebook Size

| Size | Speed | Accuracy | Notes |
|---|---|---|---|
| 16 | Fastest | Lower | May confuse acoustically similar words |
| 32 | Medium | Best balance | Recommended for 10 training samples/word |
| 64 | Slowest | Varies | May underperform with limited training data |

---

## References

- L. R. Rabiner & B.-H. Juang, *Fundamentals of Speech Recognition*, Prentice-Hall, 1993.
- Y. Linde, A. Buzo & R. M. Gray, "An algorithm for vector quantizer design," *IEEE Trans. Commun.*, 1980.
- F. Itakura, "Minimum prediction residual principle applied to speech recognition," *IEEE Trans. ASSP*, 1975.
