SAMPLE_RATE = 16000
FRAME_SIZE = 320        # 20 ms at 16 kHz
HOP_SIZE = 128          # 8 ms at 16 kHz
LPC_ORDER = 12
PRE_EMPHASIS_COEF = 0.95
CODEBOOK_SIZES = [16, 32, 64]
N_TRAIN = 10
N_TEST = 5
N_TOTAL = N_TRAIN + N_TEST  # 15 recordings per word

WORDS = [
    'start', 'stop', 'lift', 'drop', 'left',
    'right', 'forward', 'back', 'faster', 'slower',
]

RECORDINGS_DIR = 'recordings'
CODEBOOKS_DIR = 'codebooks'
RESULTS_DIR = 'results'
