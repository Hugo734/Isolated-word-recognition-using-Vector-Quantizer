"""Signal processing pipeline: pre-emphasis → framing → windowing → LPC → LSF."""
import numpy as np
import soundfile as sf
from scipy.signal import lfilter


def load_audio(path, target_sr=16000):
    audio, sr = sf.read(path)
    if sr != target_sr:
        raise ValueError(f"Expected {target_sr} Hz, got {sr} Hz in {path}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float64)


def preemphasis(signal, coef=0.95):
    """Apply Hp(z) = 1 - coef * z^{-1}."""
    return lfilter([1.0, -coef], [1.0], signal)


def frame_signal(signal, frame_size=320, hop_size=128):
    n_frames = max(1, 1 + (len(signal) - frame_size) // hop_size)
    idx = np.arange(frame_size) + np.arange(n_frames)[:, np.newaxis] * hop_size
    idx = np.minimum(idx, len(signal) - 1)
    return signal[idx]


def detect_endpoints(signal, frame_size=320, hop_size=128,
                     energy_threshold_db=20.0, margin_frames=2):
    """Return (start_sample, end_sample) of the voiced region."""
    frames = frame_signal(signal, frame_size, hop_size)
    power = np.mean(frames ** 2, axis=1)
    power_db = 10.0 * np.log10(power + 1e-10)

    floor_db = np.min(power_db)
    thresh = floor_db + energy_threshold_db

    voiced = np.where(power_db > thresh)[0]
    if len(voiced) == 0:
        return 0, len(signal)

    s = max(0, voiced[0] - margin_frames)
    e = min(len(power) - 1, voiced[-1] + margin_frames)
    return s * hop_size, min(e * hop_size + frame_size, len(signal))


def _autocorr(frame, order):
    n = len(frame)
    full = np.correlate(frame, frame, mode='full')
    return full[n - 1: n + order]   # r[0] .. r[order]


def _levinson_durbin(r, order):
    """
    Levinson-Durbin recursion.
    Returns a (LPC coeffs, length=order) and e (prediction error).
    Convention: A(z) = 1 + a[0]*z^{-1} + ... + a[p-1]*z^{-p}
    """
    a = np.zeros(order)
    e = float(r[0])

    for i in range(order):
        if e < 1e-10:
            break
        acc = np.dot(a[:i], r[i:0:-1]) if i > 0 else 0.0
        k = np.clip(-(r[i + 1] + acc) / e, -0.9999, 0.9999)

        a_prev = a.copy()
        for j in range(i):
            a[j] = a_prev[j] + k * a_prev[i - 1 - j]
        a[i] = k
        e *= 1.0 - k ** 2

    return a, max(e, 1e-10)


def lpc_analysis(frame, order=12):
    """Returns (a, gain, r): LPC coefficients, prediction error gain, autocorr."""
    r = _autocorr(frame, order)
    if r[0] < 1e-10:
        return np.zeros(order), 1.0, r
    a, e = _levinson_durbin(r, order)
    return a, e, r


def lpc_to_lsf(a):
    """
    Convert LPC coefficients to Line Spectral Frequencies (radians in [0, pi]).
    Uses companion polynomial method.  Falls back to linear spacing on failure.
    """
    order = len(a)
    A = np.concatenate([[1.0], a])          # length p+1

    A_pad = np.concatenate([A, [0.0]])      # length p+2
    A_rev = A_pad[::-1].copy()

    P_z = (A_pad + A_rev)[::-1]            # polynomial in z, symmetric
    Q_z = (A_pad - A_rev)[::-1]            # polynomial in z, antisymmetric

    # Remove trivial roots: P has z=-1, Q has z=1
    P_trim, _ = np.polydiv(P_z, [1.0, 1.0])
    Q_trim, _ = np.polydiv(Q_z, [1.0, -1.0])

    def _positive_angles(roots):
        on_unit = roots[np.abs(np.abs(roots) - 1.0) < 0.1]
        pos = on_unit[np.imag(on_unit) > -1e-6]
        angles = np.angle(pos)
        return np.sort(angles[angles > 1e-6])

    try:
        lsf = np.sort(np.concatenate([
            _positive_angles(np.roots(P_trim)),
            _positive_angles(np.roots(Q_trim)),
        ]))
    except Exception:
        lsf = np.array([])

    if len(lsf) != order:
        lsf = np.linspace(np.pi / (order + 1), np.pi * order / (order + 1), order)

    return lsf


def lpc_spectrum(a, gain, n_fft=512):
    """LPC power spectral envelope at n_fft/2+1 frequency bins."""
    A = np.zeros(n_fft)
    A[0] = 1.0
    A[1: len(a) + 1] = a
    A_fft = np.fft.rfft(A)
    return gain / (np.abs(A_fft) ** 2 + 1e-30)


def itakura_saito_dist(lpc_ref, gain_ref, lpc_test, gain_test, n_fft=512):
    """
    IS spectral distortion: how well lpc_ref explains lpc_test.
    d = mean(P_ref/P_test - log(P_ref/P_test) - 1)
    """
    P_ref = lpc_spectrum(lpc_ref, gain_ref, n_fft)
    P_test = lpc_spectrum(lpc_test, gain_test, n_fft)
    ratio = np.maximum(P_ref / (P_test + 1e-30), 1e-10)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def extract_features(signal, frame_size=320, hop_size=128,
                     lpc_order=12, pre_coef=0.95):
    """
    Full pipeline. Returns (lsf, lpc, gain) arrays or (None, None, None).
    lsf:  (n_frames, lpc_order)
    lpc:  (n_frames, lpc_order)
    gain: (n_frames,)
    """
    sig = preemphasis(signal, pre_coef)
    s, e = detect_endpoints(sig, frame_size, hop_size)
    seg = sig[s:e]

    if len(seg) < frame_size:
        return None, None, None

    frames = frame_signal(seg, frame_size, hop_size) * np.hamming(frame_size)

    lsf_list, lpc_list, gain_list = [], [], []
    for frame in frames:
        a, gain, _ = lpc_analysis(frame, lpc_order)
        lsf_list.append(lpc_to_lsf(a))
        lpc_list.append(a)
        gain_list.append(gain)

    return (np.array(lsf_list),
            np.array(lpc_list),
            np.array(gain_list))
