"""
Interactive recording script.
Run once to collect all 15 samples × 10 words at 16 kHz.
"""
import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from config import WORDS, SAMPLE_RATE, N_TOTAL, RECORDINGS_DIR

DURATION = 2.0  # seconds per utterance


def _record():
    print(f"  >>> Recording {DURATION}s — speak now! <<<")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()


def main():
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    print("=== Speech Dataset Collection ===")
    print(f"{N_TOTAL} samples × {len(WORDS)} words at {SAMPLE_RATE} Hz.")
    print("Quiet room, same speaker throughout.\n")

    for word in WORDS:
        word_dir = os.path.join(RECORDINGS_DIR, word)
        os.makedirs(word_dir, exist_ok=True)

        print(f"\n{'='*44}")
        print(f"   Word: '{word.upper()}'")
        print(f"{'='*44}")
        input("   Press Enter when ready...")

        for i in range(1, N_TOTAL + 1):
            path = os.path.join(word_dir, f'{i:02d}.wav')
            if os.path.exists(path):
                ans = input(f"   [{i:02d}/{N_TOTAL}] Already recorded. Re-record? [y/N]: ").strip().lower()
                if ans != 'y':
                    continue

            input(f"   [{i:02d}/{N_TOTAL}] Press Enter to record '{word}'...")
            time.sleep(0.15)
            audio = _record()
            sf.write(path, audio, SAMPLE_RATE)
            print(f"   Saved → {path}")
            time.sleep(0.4)

    print("\nAll recordings collected.")


if __name__ == '__main__':
    main()
