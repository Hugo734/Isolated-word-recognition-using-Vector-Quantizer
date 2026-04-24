"""Recognition stage: classify test utterances and compute confusion matrices."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import (WORDS, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE, LPC_ORDER,
                    PRE_EMPHASIS_COEF, CODEBOOK_SIZES, N_TRAIN, N_TOTAL,
                    RECORDINGS_DIR, CODEBOOKS_DIR, RESULTS_DIR)
from features import load_audio, extract_features, itakura_saito_dist


def _load_codebook(word, size):
    path = os.path.join(CODEBOOKS_DIR, f'{word}_{size}.npz')
    d = np.load(path)
    return d['lpc'], d['gain']   # (K, 12), (K,)


def _recognize(lpc_frames, gain_frames, codebooks):
    """
    Classify using minimum average IS distortion.
    codebooks: list of (cb_lpc, cb_gain) per word.
    Returns predicted word index.
    """
    best_word, best_score = 0, np.inf
    for idx, (cb_lpc, cb_gain) in enumerate(codebooks):
        total = 0.0
        for lpc_t, gain_t in zip(lpc_frames, gain_frames):
            total += min(
                itakura_saito_dist(cb_lpc[j], cb_gain[j], lpc_t, gain_t)
                for j in range(len(cb_lpc))
            )
        score = total / max(len(lpc_frames), 1)
        if score < best_score:
            best_score, best_word = score, idx
    return best_word


def _plot(confusion, size, accuracy):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion, cmap='Blues', vmin=0)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(WORDS)))
    ax.set_yticks(range(len(WORDS)))
    ax.set_xticklabels(WORDS, rotation=45, ha='right')
    ax.set_yticklabels(WORDS)
    for i in range(len(WORDS)):
        for j in range(len(WORDS)):
            color = 'white' if confusion[i, j] > confusion.max() / 2 else 'black'
            ax.text(j, i, str(confusion[i, j]),
                    ha='center', va='center', color=color, fontsize=10)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — Codebook size {size} | Accuracy: {accuracy:.1%}')
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f'confusion_{size}.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def evaluate(codebook_size):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        codebooks = [_load_codebook(w, codebook_size) for w in WORDS]
    except FileNotFoundError as exc:
        print(f"Missing codebook ({exc}). Run train.py first.")
        return None, None

    n = len(WORDS)
    confusion = np.zeros((n, n), dtype=int)

    for true_idx, word in enumerate(WORDS):
        word_dir = os.path.join(RECORDINGS_DIR, word)
        for i in range(N_TRAIN + 1, N_TOTAL + 1):
            path = os.path.join(word_dir, f'{i:02d}.wav')
            if not os.path.exists(path):
                continue
            sig = load_audio(path, SAMPLE_RATE)
            _, lpc, gain = extract_features(sig, FRAME_SIZE, HOP_SIZE,
                                            LPC_ORDER, PRE_EMPHASIS_COEF)
            if lpc is None:
                continue
            pred = _recognize(lpc, gain, codebooks)
            confusion[true_idx, pred] += 1

    total = confusion.sum()
    accuracy = np.trace(confusion) / total if total else 0.0

    print(f"\n=== Codebook size {codebook_size} | Accuracy: {accuracy:.1%} ===")
    header = ''.join(f'{w:>9}' for w in WORDS)
    print(f"{'':>9}{header}")
    for i, row in enumerate(confusion):
        print(f'{WORDS[i]:>9}' + ''.join(f'{v:>9}' for v in row))

    _plot(confusion, codebook_size, accuracy)
    return confusion, accuracy


if __name__ == '__main__':
    for size in CODEBOOK_SIZES:
        evaluate(size)
