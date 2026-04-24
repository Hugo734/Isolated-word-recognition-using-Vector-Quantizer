# Práctica 1 — Reconocimiento de Palabras con Cuantización Vectorial

Speech recognition system using Vector Quantization (VQ) and LPC features. Records 10 words × 15 times, trains codebooks with the LBG algorithm, and evaluates recognition accuracy via confusion matrices.

---

## Requirements

- Python 3.10+
- A microphone connected to your computer

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Step 1 — record your voice
python main.py record

# Step 2 — train the codebooks
python main.py train

# Step 3 — evaluate and print confusion matrices
python main.py evaluate
```

Or run everything after recording:

```bash
python main.py all
```

---

## Words Used

The 10 words configured for the Puzzlebot navigation challenge:

| # | Word | # | Word |
|---|------|---|------|
| 1 | start | 6 | right |
| 2 | stop | 7 | forward |
| 3 | lift | 8 | back |
| 4 | drop | 9 | faster |
| 5 | left | 10 | slower |

To change the words, edit `WORDS` in `config.py`.

---

## Recording Instructions

Run:

```bash
python main.py record
```

The script will guide you word by word. For each word:

1. It shows the word in uppercase and waits for you to press **Enter**.
2. It then asks you to press **Enter** again before each of the 15 recordings.
3. Each recording lasts **2 seconds** — say the word clearly once.
4. Files are saved automatically under `recordings/<word>/01.wav … 15.wav`.

**Tips for good recordings:**
- Use a quiet room with no background noise.
- Keep the same speaker for all 150 recordings.
- Say the word once, naturally, in the middle of the 2-second window.
- Keep the same distance from the microphone throughout.
- If a recording goes wrong, the script asks if you want to overwrite it.

**File layout after recording:**

```
recordings/
├── start/
│   ├── 01.wav
│   ├── 02.wav
│   └── ... (up to 15.wav)
├── stop/
│   └── ...
└── ...
```

---

## Training

```bash
python main.py train
```

Uses the first **10 recordings** of each word to train three codebooks (sizes 16, 32, 64).

What happens internally:

1. Each `.wav` is loaded at 16 kHz.
2. A pre-emphasis filter `H(z) = 1 − 0.95z⁻¹` is applied.
3. The voiced segment is detected using frame energy (20 dB threshold above the noise floor).
4. The signal is split into **320-sample Hamming-windowed frames** with a **128-sample hop**.
5. For each frame, **LPC order 12** coefficients are computed via the autocorrelation method and Levinson-Durbin recursion.
6. The LPC coefficients are converted to **Line Spectral Frequencies (LSF)**.
7. All LSF vectors from all 10 training files are pooled and clustered with the **LBG algorithm** to produce codebooks of size 16, 32, and 64.
8. For each codebook entry, the mean LPC coefficients and gain of its cluster members are stored (used later for the Itakura-Saito distance).

Codebooks are saved under `codebooks/`:

```
codebooks/
├── start_16.npz
├── start_32.npz
├── start_64.npz
├── stop_16.npz
└── ...
```

---

## Evaluation

```bash
python main.py evaluate          # all three sizes
python main.py evaluate 32       # only size 32
python main.py evaluate 16 64    # sizes 16 and 64
```

Uses the last **5 recordings** (files `11.wav` to `15.wav`) of each word.

For each test utterance:

1. Features are extracted exactly as in training.
2. For every frame, the **Itakura-Saito (IS) spectral distortion** is computed against every codeword in each word's codebook.
3. The minimum IS distance across all codewords gives the frame's contribution to that word's score.
4. The word with the **lowest average IS distortion** across all frames is the predicted label.
5. Results are accumulated into a **10×10 confusion matrix**.

Output printed to console:

```
=== Codebook size 32 | Accuracy: 84.0% ===
Confusion matrix (rows=true, cols=predicted):
          start     stop     lift     drop ...
    start     5        0        0        0 ...
     stop     0        4        1        0 ...
     ...
```

Confusion matrix plots saved to `results/`:

```
results/
├── confusion_16.png
├── confusion_32.png
└── confusion_64.png
```

---

## File Reference

| File | Purpose |
|---|---|
| `config.py` | All constants: sample rate, frame size, hop size, LPC order, codebook sizes, words, paths |
| `features.py` | Signal processing: pre-emphasis, framing, VAD, LPC analysis, LSF conversion, IS distance |
| `vq.py` | LBG vector quantization algorithm |
| `train.py` | Loads training audio, extracts features, runs LBG, saves codebooks |
| `recognize.py` | Loads test audio, classifies with IS distance, builds and plots confusion matrix |
| `record_words.py` | Interactive CLI for recording all 150 utterances |
| `main.py` | Unified entry point (`record`, `train`, `evaluate`, `all`) |

---

## Configuration (`config.py`)

| Parameter | Value | Description |
|---|---|---|
| `SAMPLE_RATE` | 16000 Hz | Recording and analysis sample rate |
| `FRAME_SIZE` | 320 samples | 20 ms Hamming window |
| `HOP_SIZE` | 128 samples | 8 ms frame shift |
| `LPC_ORDER` | 12 | LPC analysis order |
| `PRE_EMPHASIS_COEF` | 0.95 | Pre-emphasis filter coefficient |
| `CODEBOOK_SIZES` | [16, 32, 64] | VQ codebook sizes to train and compare |
| `N_TRAIN` | 10 | Recordings per word used for training |
| `N_TEST` | 5 | Recordings per word used for testing |

---

## Signal Processing Pipeline

```
Raw audio (16 kHz)
       │
       ▼
Pre-emphasis:  H(z) = 1 − 0.95·z⁻¹
       │
       ▼
Voice Activity Detection (energy threshold, 20 dB above noise floor)
       │
       ▼
Framing: 320-sample Hamming windows, 128-sample hop
       │
       ▼
LPC analysis (order 12): autocorrelation + Levinson-Durbin recursion
       │
       ▼
LSF conversion: companion polynomial root-finding
       │  (used for VQ training — Euclidean distance)
       ▼
LBG clustering → codebook of size 16 / 32 / 64
       │  (stored alongside mean LPC per cluster for recognition)
       ▼
Recognition: Itakura-Saito spectral distortion (computed via FFT)
       │
       ▼
Confusion matrix
```

---

## Notes on Codebook Size

Larger codebooks capture more detail per word but require more training data to fill reliably. Expected behaviour:

- **Size 16** — fastest, lower accuracy, may confuse acoustically similar words.
- **Size 32** — good balance for 10 training files per word.
- **Size 64** — highest capacity; may underperform if training data is insufficient.

Compare the three confusion matrices to decide which size works best for your recordings.
# Isolated-word-recognition-using-Vector-Quantizer
