"""Train one VQ codebook per word per codebook size."""
import os
import numpy as np
from config import (WORDS, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE, LPC_ORDER,
                    PRE_EMPHASIS_COEF, CODEBOOK_SIZES, N_TRAIN,
                    RECORDINGS_DIR, CODEBOOKS_DIR)
from features import load_audio, extract_features
from vq import lbg, assign_clusters


def train_all_codebooks():
    os.makedirs(CODEBOOKS_DIR, exist_ok=True)

    for word in WORDS:
        print(f"\n[{word}] Extracting features from {N_TRAIN} training files...")
        word_dir = os.path.join(RECORDINGS_DIR, word)

        all_lsf, all_lpc, all_gain = [], [], []
        for i in range(1, N_TRAIN + 1):
            path = os.path.join(word_dir, f'{i:02d}.wav')
            if not os.path.exists(path):
                print(f"  Missing: {path}")
                continue
            sig = load_audio(path, SAMPLE_RATE)
            lsf, lpc, gain = extract_features(sig, FRAME_SIZE, HOP_SIZE,
                                               LPC_ORDER, PRE_EMPHASIS_COEF)
            if lsf is not None:
                all_lsf.append(lsf)
                all_lpc.append(lpc)
                all_gain.append(gain)

        if not all_lsf:
            print(f"  No valid data for '{word}', skipping.")
            continue

        lsf_mat = np.vstack(all_lsf)        # (N_total_frames, 12)
        lpc_mat = np.vstack(all_lpc)        # (N_total_frames, 12)
        gain_vec = np.concatenate(all_gain)  # (N_total_frames,)

        print(f"  {len(lsf_mat)} frames total.")

        for size in CODEBOOK_SIZES:
            print(f"  LBG size={size}...", end=' ', flush=True)
            cb_lsf = lbg(lsf_mat, size)

            # Compute mean LPC and gain for each cluster (for IS distance later)
            asgn = assign_clusters(lsf_mat, cb_lsf)
            cb_lpc = np.zeros((size, LPC_ORDER))
            cb_gain = np.zeros(size)
            for j in range(size):
                mask = asgn == j
                if mask.any():
                    cb_lpc[j] = lpc_mat[mask].mean(axis=0)
                    cb_gain[j] = gain_vec[mask].mean()
                else:
                    # Empty cluster: use global mean
                    cb_lpc[j] = lpc_mat.mean(axis=0)
                    cb_gain[j] = gain_vec.mean()

            out = os.path.join(CODEBOOKS_DIR, f'{word}_{size}.npz')
            np.savez(out, lsf=cb_lsf, lpc=cb_lpc, gain=cb_gain)
            print(f"saved {out}")


if __name__ == '__main__':
    train_all_codebooks()
