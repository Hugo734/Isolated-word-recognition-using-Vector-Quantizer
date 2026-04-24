#!/usr/bin/env python3
"""
Práctica 1 — Reconocimiento de Palabras con Cuantización Vectorial

Usage:
  python main.py record              collect 15 recordings × 10 words
  python main.py train               train VQ codebooks (sizes 16, 32, 64)
  python main.py evaluate [size...]  compute confusion matrices
  python main.py all                 train + evaluate all sizes
"""
import sys


def _cmd_record():
    from record_words import main as rec
    rec()


def _cmd_train():
    from train import train_all_codebooks
    train_all_codebooks()


def _cmd_evaluate(sizes):
    from recognize import evaluate
    for s in sizes:
        evaluate(s)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()
    from config import CODEBOOK_SIZES

    if cmd == 'record':
        _cmd_record()
    elif cmd == 'train':
        _cmd_train()
    elif cmd == 'evaluate':
        sizes = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else CODEBOOK_SIZES
        _cmd_evaluate(sizes)
    elif cmd == 'all':
        _cmd_train()
        _cmd_evaluate(CODEBOOK_SIZES)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()
