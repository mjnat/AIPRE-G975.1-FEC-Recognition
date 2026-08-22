#!/usr/bin/env python3
"""
Sovereign Reconstruction — RRC Pulse Shaping + Syndrome Kernel + Hybrid Features
==================================================================================
1. VERIFY RRC: Eye diagram with RRC roll-off 0.2, 0 dBm
2. TRUTH TEST: SSFM + CDC + RRC @ 32dB, hybrid features (HOS + Syndrome + 2D CNN)
3. EVALUATE: Per-class classification with confusion matrix
"""

import os, sys, warnings, gc, time
sys.stdout.reconfigure(line_buffering=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# RTX 3080 Ti — preload CUDA 11.8 libs for TF 2.13
import ctypes
_nvidia_lib = os.path.expanduser('~/.local/lib/python3.8/site-packages/nvidia')
_load_map = [
    ('cuda_runtime', 'libcudart.so.11.0'),
    ('cublas',       'libcublas.so.11'),
    ('cudnn',        'libcudnn.so.8'),
    ('cufft',        'libcufft.so.10'),
    ('curand',       'libcurand.so.10'),
    ('cusolver',     'libcusolver.so.11'),
    ('cusparse',     'libcusparse.so.11'),
]
for _pkg, _soname in _load_map:
    _lib_path = os.path.join(_nvidia_lib, _pkg, 'lib', _soname)
    if os.path.exists(_lib_path):
        try:
            ctypes.CDLL(_lib_path)
        except OSError:
            pass

warnings.filterwarnings('ignore')

import dataset_g975_1_fixed as dataset_mod
import galois
import h5py
import numpy as np
import tensorflow as tf
tf.keras.backend.clear_session()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras import layers, models, callbacks, regularizers

# Enable GPU memory growth (don't grab all VRAM at once)
_gpus = tf.config.list_physical_devices('GPU')
for _g in _gpus:
    tf.config.experimental.set_memory_growth(_g, True)

# Mixed precision — FP16 on Tensor Cores (3x speedup on Ampere)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import skew, kurtosis, entropy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

rng = np.random.default_rng(42)
import numba
from numba import njit, prange
BATCH_SIZE = 256
N_CLASSES = 9
SPS = dataset_mod.SPS
SAMPLING_RATE = dataset_mod.SAMPLING_RATE
AI_WINDOW = dataset_mod.AI_WINDOW
N_SYMBOLS = AI_WINDOW // SPS  # 2048
FEC_MATRIX_SHAPE = (32, 64)

# ==============================================================================
# Galois fields for PCM construction
# ==============================================================================
GF2 = galois.GF2
GF256 = dataset_mod.GF256
GF10  = dataset_mod.GF10
GF11  = dataset_mod.GF11
GF12  = dataset_mod.GF12

# ==============================================================================
# FWHT helpers
# ==============================================================================
@njit(cache=True)
def custom_fwht(x):
    n = len(x)
    next_pow2 = 1 << (n - 1).bit_length() if n > 0 else 1
    padded = np.zeros(next_pow2, dtype=np.float64)
    padded[:n] = x
    h = 1
    while h < next_pow2:
        for i in range(0, next_pow2, h * 2):
            for j in range(i, i + h):
                a = padded[j]; b = padded[j + h]
                padded[j] = a + b; padded[j + h] = a - b
        h <<= 1
    return padded

@njit(parallel=True, cache=True)
def custom_fwht_2d(matrix):
    rows, cols = matrix.shape
    nr = 1 << (rows.bit_length())
    nc = 1 << (cols.bit_length())
    padded = np.zeros((nr, nc), dtype=np.float64)
    padded[:rows, :cols] = matrix
    for i in prange(nr):
        padded[i, :] = custom_fwht(padded[i, :])
    for j in prange(nc):
        padded[:, j] = custom_fwht(padded[:, j])
    return padded

# ==============================================================================
# I.5 Hamming PCM for syndrome features
# ==============================================================================
def _get_hamming_pcm(n):
    m = int(np.ceil(np.log2(n + 1)))
    h = []
    for i in range(1, n + 1):
        h.append([int(x) for x in format(i, f'0{m}b')])
    return np.array(h, dtype=np.uint8).T

_I5_COL_PCM = _get_hamming_pcm(512)   # 9×512 — checks 512-bit column codewords
_I5_ROW_PCM = _get_hamming_pcm(510)   # 9×510 — checks 510-bit row codewords

# ==============================================================================
# HOS Feature Extraction (replaces old AV features)
# ==============================================================================
def extract_hos_features(iq_block_sym):
    """Extract HOS + entropy features from symbol-spaced IQ (N_sym, 2).
    Returns 16 features: [mean, var, skew, kurt, FWHT_mean, FWHT_var,
                          FWHT_skew, FWHT_kurt, energy_conc, peak_count,
                          entropy, entropy_residual,
                          bit_matrix_rank, WHT2D_mean, WHT2D_var, WHT2D_skew, WHT2D_kurt]
    """
    feats = np.zeros(17, dtype=np.float32)
    try:
        I = iq_block_sym[:, 0].astype(np.float64)
        feats[0] = np.mean(I)
        feats[1] = np.var(I)
        feats[2] = skew(I) if len(I) > 2 else 0.0
        feats[3] = kurtosis(I) if len(I) > 3 else 0.0

        n = len(I)
        next_pow2 = 1 << (n - 1).bit_length() if n > 0 else 1
        padded = np.pad(I, (0, max(0, next_pow2 - n)), 'constant')
        spec = np.abs(custom_fwht(padded)[1:])
        if len(spec) > 10:
            feats[4] = np.mean(spec)
            feats[5] = np.var(spec)
            feats[6] = skew(spec)
            feats[7] = kurtosis(spec)
            total_e = np.sum(spec**2)
            feats[8] = np.sum(np.sort(spec)[::-1][:max(1, int(len(spec)*0.05))]**2) / max(total_e, 1e-15)
            feats[9] = float(np.sum(spec > np.mean(spec) + 3*np.std(spec)))

        p_dist, _ = np.histogram(I, bins='auto', density=True)
        p_dist = p_dist[p_dist > 0]
        feats[10] = entropy(p_dist, base=2) if len(p_dist) > 1 else 0.0
        feats[11] = abs(feats[10] - np.log2(max(len(p_dist), 1))) if len(p_dist) > 0 else 0.0

        # Bit matrix features (WHT 2D)
        bits = (I > 0).astype(np.float32)
        n_sym = FEC_MATRIX_SHAPE[0] * FEC_MATRIX_SHAPE[1]
        if len(bits) >= n_sym:
            bits = bits[:n_sym]
        else:
            bits = np.pad(bits, (0, n_sym - len(bits)), 'constant')
        matrix = bits.reshape(FEC_MATRIX_SHAPE)
        feats[12] = float(np.linalg.matrix_rank(matrix.astype(np.float32)))
        wht2d = np.abs(custom_fwht_2d(matrix))
        flat = wht2d.flatten()
        feats[13] = np.mean(flat)
        feats[14] = np.var(flat)
        feats[15] = skew(flat) if len(flat) > 2 else 0.0
        feats[16] = kurtosis(flat) if len(flat) > 3 else 0.0
    except:
        pass
    return feats

def batch_extract_hos(X_batch_sym):
    n = X_batch_sym.shape[0]
    feats = np.zeros((n, 17), dtype=np.float32)
    for i in range(n):
        feats[i] = extract_hos_features(X_batch_sym[i])
    return feats

# ==============================================================================
# Syndrome Kernel — I.5 product code syndrome weights
# ==============================================================================
@njit(cache=True)
def compute_i5_syndrome(bits_hard, pcm):
    """I.5 syndrome weight features: [mean_weight, var_weight] across rows."""
    n_cols = pcm.shape[1]
    n_avail = len(bits_hard)
    if n_avail < n_cols:
        return np.array([0.0, 0.0], dtype=np.float32)
    n_rows = n_avail // n_cols
    if n_rows < 1:
        return np.array([0.0, 0.0], dtype=np.float32)
    matrix = bits_hard[:n_rows * n_cols].reshape((n_rows, n_cols))
    syn = (matrix.astype(np.float64) @ pcm.astype(np.float64).T) % 2
    weights = np.sum(syn, axis=1)
    return np.array([np.mean(weights), np.var(weights)], dtype=np.float32)

def compute_i7_syndrome(bits_hard):
    """I.7 syndrome weight features using row BCH generator polynomial."""
    try:
        n_cols = 900
        n_avail = len(bits_hard)
        if n_avail < n_cols * 2:
            return np.array([0.0, 0.0], dtype=np.float32)
        n_rows = n_avail // n_cols
        matrix = bits_hard[:n_rows * n_cols].astype(np.uint8).reshape((n_rows, n_cols))
        g_row = dataset_mod.get_bch_standard_polynomials()[3]
        g_row_len = len(g_row)
        weights = np.zeros(n_rows, dtype=np.float64)
        for i in range(n_rows):
            r = gf2_poly_div(matrix[i], g_row)
            weights[i] = np.sum(r)
        return np.array([np.mean(weights), np.var(weights)], dtype=np.float32)
    except:
        return np.array([0.0, 0.0], dtype=np.float32)

@njit(cache=True)
def gf2_poly_div(a, g):
    r = a.copy().astype(np.uint8)
    for i in range(len(a) - len(g) + 1):
        if r[i] == 1:
            for j in range(len(g)):
                r[i + j] ^= g[j]
    return r[-(len(g) - 1):]

def extract_syndrome_features(iq_block_sym):
    """Syndrome kernel: 4 features = [I5_mean, I5_var, I7_mean, I7_var]."""
    I = iq_block_sym[:, 0]
    bits = (I > 0).astype(np.uint8)
    i5 = compute_i5_syndrome(bits, _I5_COL_PCM)
    i7 = compute_i7_syndrome(bits)
    return np.concatenate([i5, i7]).astype(np.float32)

def batch_extract_syndrome(X_batch_sym):
    n = X_batch_sym.shape[0]
    feats = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        feats[i] = extract_syndrome_features(X_batch_sym[i])
    return feats

@njit(parallel=True, cache=True)
def batch_syndrome_jit(X_batch_sym, pcm):
    """Parallel syndrome extraction for I.5 only (the heavy part)."""
    n = X_batch_sym.shape[0]
    feats = np.zeros((n, 2), dtype=np.float32)
    for i in prange(n):
        I = X_batch_sym[i, :, 0]
        bits = (I > 0).astype(np.uint8)
        feats[i] = compute_i5_syndrome(bits, pcm)
    return feats

# ==============================================================================
# Bit Matrix preparation for 2D CNN
# ==============================================================================
def extract_bit_matrix(iq_block_sym):
    """Convert symbol-spaced IQ to (32, 64) binary matrix for 2D CNN."""
    I = iq_block_sym[:, 0]
    bits = (I > 0).astype(np.float32)
    n_needed = FEC_MATRIX_SHAPE[0] * FEC_MATRIX_SHAPE[1]
    if len(bits) >= n_needed:
        bits = bits[:n_needed]
    else:
        bits = np.pad(bits, (0, n_needed - len(bits)), 'constant')
    return bits.reshape(FEC_MATRIX_SHAPE)

def batch_extract_bit_matrix(X_batch_sym):
    n = X_batch_sym.shape[0]
    mat = np.zeros((n, *FEC_MATRIX_SHAPE), dtype=np.float32)
    for i in range(n):
        mat[i] = extract_bit_matrix(X_batch_sym[i])
    return mat[..., np.newaxis]  # (N, 32, 64, 1)

# ==============================================================================
# 2D CNN + Features Hybrid Model
# ==============================================================================
def build_sovereign_model(feature_dim, n_classes=N_CLASSES):
    """Hybrid model: 2D CNN on bit matrix + Dense branch on feature vector."""
    cnn_input = layers.Input(shape=(*FEC_MATRIX_SHAPE, 1), name='bit_matrix')
    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False)(cnn_input)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(384, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)

    feat_input = layers.Input(shape=(feature_dim,), name='features')
    y = layers.Dense(128, activation='relu')(feat_input)
    y = layers.Dense(64, activation='relu')(y)

    z = layers.Concatenate()([x, y])
    z = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(z)
    z = layers.Dropout(0.4)(z)
    z = layers.Dense(256, activation='relu')(z)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(128, activation='relu')(z)
    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32')(z)

    model = models.Model(inputs=[cnn_input, feat_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==============================================================================
# Training Loop — Sovereign Reconstruction
# ==============================================================================
def train_eval_sovereign(X_bit_mat, X_feat, y, label_str="", epochs=200, batch_size=BATCH_SIZE):
    idx_by_class = {c: np.where(y == c)[0] for c in range(N_CLASSES)}
    train_idx, test_idx = [], []
    for c in range(N_CLASSES):
        perm = rng.permutation(idx_by_class[c])
        tr, te = train_test_split(perm, test_size=0.2, random_state=c)
        train_idx.append(tr); test_idx.append(te)
    train_idx = np.sort(np.concatenate(train_idx)).astype(np.int64)
    test_idx = np.sort(np.concatenate(test_idx)).astype(np.int64)

    Xm_train = X_bit_mat[train_idx]; Xm_test = X_bit_mat[test_idx]
    Xf_train = X_feat[train_idx]; Xf_test = X_feat[test_idx]
    y_train = y[train_idx]; y_test = y[test_idx]
    print(f"  Train: {len(train_idx)}  Test: {len(test_idx)}")

    scaler = StandardScaler()
    Xf_train = scaler.fit_transform(Xf_train).astype(np.float32)
    Xf_test = scaler.transform(Xf_test).astype(np.float32)

    y_train_cat = tf.keras.utils.to_categorical(y_train, N_CLASSES)
    y_test_cat = tf.keras.utils.to_categorical(y_test, N_CLASSES)

    model = build_sovereign_model(X_feat.shape[1], N_CLASSES)
    cb_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=30,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=15, min_lr=1e-6, verbose=1),
    ]

    # tf.data pipeline — cache + prefetch for zero-wait GPU feeding
    train_ds = tf.data.Dataset.from_tensor_slices(
        ({'bit_matrix': Xm_train, 'features': Xf_train}, y_train_cat))
    train_ds = train_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        ({'bit_matrix': Xm_test, 'features': Xf_test}, y_test_cat))
    val_ds = val_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"\nTraining {label_str} ...")
    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=epochs, callbacks=cb_list, verbose=1)
    t_train = time.time() - t0
    n_epochs_done = len(history.history['accuracy'])
    imgs_per_sec = len(train_idx) * n_epochs_done / max(t_train, 1e-6)
    print(f"\n  >>> IMAGES/SEC: {imgs_per_sec:.1f} | "
          f"Train time: {t_train:.1f}s | Epochs: {n_epochs_done} <<<")

    y_prob = model.predict([Xm_test, Xf_test], verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    test_acc = np.mean(y_pred == y_test)
    print(f"\n>>> {label_str} TEST ACCURACY: {test_acc * 100:.2f}% <<<")

    target_names = [
        "I.1 (RS 239/255)", "I.2 (RS+CSOC)", "I.3 (BCH concat)",
        "I.4 (RS+BCH inter)", "I.5 (RS product)", "I.6 (LDPC stair)",
        "I.7 (BCH product)", "I.8 (RS GF12)", "I.9 (BCH dual)"
    ]
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"I.{i+1}" for i in range(9)],
                yticklabels=[f"I.{i+1}" for i in range(9)], ax=axes[0])
    axes[0].set_title(f"{label_str} Confusion Matrix\nTest Acc: {test_acc*100:.1f}%")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=[f"I.{i+1}" for i in range(9)],
                yticklabels=[f"I.{i+1}" for i in range(9)], ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized per class)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    plt.suptitle(f"Sovereign Reconstruction — {label_str}", fontsize=13, fontweight='bold')
    sfx = label_str.lower().replace(" ", "_").replace("-", "")
    plt.tight_layout()
    plt.savefig(f"confusion_{sfx}.png", dpi=300, bbox_inches='tight')
    print(f"  confusion_{sfx}.png saved")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    axes2[0].plot(history.history['accuracy'], label='Train')
    axes2[0].plot(history.history['val_accuracy'], label='Val')
    axes2[0].set_title('Accuracy'); axes2[0].legend(); axes2[0].grid()
    axes2[1].plot(history.history['loss'], label='Train')
    axes2[1].plot(history.history['val_loss'], label='Val')
    axes2[1].set_title('Loss'); axes2[1].legend(); axes2[1].grid()
    plt.suptitle(f"Sovereign Reconstruction — {label_str} Training History")
    plt.tight_layout()
    plt.savefig(f"training_{sfx}.png", dpi=150)
    print(f"  training_{sfx}.png saved")

    print(classification_report(y_test, y_pred, labels=list(range(9)),
                                target_names=target_names, digits=4))
    return test_acc

# ==============================================================================
# VERIFY RRC — Eye Diagram with RRC pulse shaping
# ==============================================================================
def rrc_eye_diagram():
    print("\n" + "=" * 70)
    print("VERIFY RRC: Eye Diagram — RRC roll-off 0.2, 0dBm launch")
    print("=" * 70)

    raw_bits = np.random.randint(0, 2, 1000000, dtype=np.uint8)
    GF256 = galois.GF(2**8, irreducible_poly=0x11d)
    rs = galois.ReedSolomon(255, 239, field=GF256, c=0)
    syms_array = dataset_mod.bits_to_uints_gf_exact(raw_bits[:239*8], 8)
    bits = np.array([[(s >> j) & 1 for j in range(7, -1, -1)]
                     for s in rs.encode(syms_array)]).flatten()
    segment = np.resize(bits, AI_WINDOW)
    complex_sig = (segment.astype(np.complex128)*2 - 1) * np.sqrt(dataset_mod.P_LAUNCH_W)

    oversampled = dataset_mod._pulse_shaping_filter(complex_sig, SPS)
    n_symbols = len(complex_sig)
    n = len(oversampled)

    # SSFM pipeline (same as _fixed)
    dt = 1.0 / SAMPLING_RATE
    frequencies = np.fft.fftfreq(n, d=dt)
    beta2 = -2.1e-26; alpha = 4.6e-5
    distance_m = 80000.0; num_steps = 40
    h_step = distance_m / num_steps
    h_eff = (1.0 - np.exp(-alpha * h_step)) / alpha
    channel_spacing_hz = 100e9
    walkoff_delay_sec = np.abs(beta2 * 2.0 * np.pi * channel_spacing_hz * h_step)
    walkoff_shift_per_step = walkoff_delay_sec * SAMPLING_RATE
    dispersion_half_step = np.exp(-1j * 0.5 * beta2 * (2.0 * np.pi * frequencies)**2 * (h_step / 2.0))

    oversampled_xpm_sum = np.zeros(n, dtype=np.float64)
    for _ in range(4):
        bits_ch = np.random.randint(0, 4, n_symbols, dtype=np.uint8)
        sig_ch = (np.cos(bits_ch * np.pi / 2) + 1j * np.sin(bits_ch * np.pi / 2)) * np.sqrt(dataset_mod.P_LAUNCH_W)
        interferer = dataset_mod._pulse_shaping_filter(sig_ch, SPS)
        oversampled_xpm_sum += np.abs(interferer)**2
    u1 = oversampled.copy()
    for step in range(num_steps):
        u1 = np.fft.ifft(np.fft.fft(u1) * dispersion_half_step)
        power_spm = np.abs(u1)**2
        shift_samples = int(step * walkoff_shift_per_step)
        power_xpm = dataset_mod.roll_array_1d(oversampled_xpm_sum, shift_samples) * np.exp(-alpha * step * h_step)
        phi_nl = dataset_mod.GAMMA * h_eff * (power_spm + 2.0 * power_xpm)
        u1 = u1 * (np.cos(phi_nl) + 1j * np.sin(phi_nl))
        u1 = np.fft.ifft(np.fft.fft(u1) * dispersion_half_step)
        u1 *= np.exp(-0.5 * alpha * h_step)

    # ASE at 32dB using received power
    ref_bw = 12.5e9
    P_rx = np.mean(np.abs(u1)**2)
    N0 = P_rx / (2.0 * 10**(32 / 10.0) * ref_bw)
    noise_std = np.sqrt(N0 * SAMPLING_RATE / 2.0)
    noise = (np.random.normal(0, noise_std, n) + 1j * np.random.normal(0, noise_std, n))
    received = u1 + noise

    # CDC
    omega = 2.0 * np.pi * frequencies
    cdc_transfer = np.exp(1j * 0.5 * beta2 * omega**2 * distance_m)
    received = np.fft.ifft(np.fft.fft(received) * cdc_transfer)

    # RRC matched filter
    mf_full = dataset_mod._pulse_shaping_filter(received, SPS)
    I_full = np.real(mf_full)
    total_symbols = len(I_full) // SPS

    # Eye diagram
    fig, ax = plt.subplots(figsize=(12, 6))
    n_segments = 300
    for _ in range(n_segments):
        sym_start = np.random.randint(0, total_symbols - 3)
        idx_start = sym_start * SPS
        idx_end = idx_start + 3 * SPS
        seg = I_full[idx_start:idx_end]
        t_rel = np.arange(len(seg))
        ax.plot(t_rel, seg, 'b-', alpha=0.15, linewidth=0.5)
    ax.set_xlabel('Sample offset', fontsize=12)
    ax.set_ylabel('Amplitude (I)', fontsize=12)
    ax.set_title(f'RRC Eye Diagram — Class 0, 32dB OSNR, 0dBm\n'
                 f'Roll-off {dataset_mod.RRC_ALPHA}, SPS={SPS}, {n_segments} traces',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    for s in range(0, 4 * SPS + 1, SPS):
        ax.axvline(s, color='red', linestyle='--', alpha=0.2)
    ax.axvline(SPS//2, color='green', linestyle='-', alpha=0.8, label=f'Sampling @ idx {SPS//2}')
    ax.legend()
    plt.tight_layout()
    plt.savefig("eye_diagram_rrc.png", dpi=300, bbox_inches='tight')
    print("\neye_diagram_rrc.png saved")

    # Symbol-spaced statistics
    n_avg = min(5000, total_symbols)
    symbol_wfs = np.zeros((n_avg, SPS))
    for i in range(n_avg):
        symbol_wfs[i] = I_full[i * SPS : (i+1) * SPS]
    std_wf = np.std(symbol_wfs, axis=0)
    print(f"\n  Symbol-spaced std (SPS={SPS}):")
    for i in range(SPS):
        tag = " <-- CURRENT" if i == SPS//2 else ""
        print(f"    Sample {i}: std={std_wf[i]:.6f}{tag}")
    print(f"\n  Eye opening metric (min std / max std): {std_wf.min()/max(std_wf.max(),1e-15):.4f}")
    return std_wf

# ==============================================================================
# SOVEREIGN RECONSTRUCTION TEST
# ==============================================================================
def sovereign_test():
    print("=" * 70)
    print("SOVEREIGN RECONSTRUCTION — SSFM + CDC + RRC @ 32dB OSNR")
    print("=" * 70)

    N_SAMPLES_PER_CLASS = 1000
    total = N_CLASSES * N_SAMPLES_PER_CLASS
    print(f"\nLoading {total} precomputed samples ({N_SAMPLES_PER_CLASS}/class @ 32dB) ...")

    with h5py.File('dataset_g975_bitexact_sovereign_final.h5', 'r') as f:
        X_full = f['X'][:]   # Load entire dataset into RAM
        y_full = f['y'][:]
    print(f"  Dataset: {X_full.shape}")
    X_raw = np.zeros((total, AI_WINDOW, 2), dtype=np.float32)
    y_raw = np.zeros(total, dtype=np.int32)
    ptr = 0
    for c in range(N_CLASSES):
        idx = np.where(y_full == c)[0]
        chosen = rng.choice(idx, N_SAMPLES_PER_CLASS, replace=False)
        for i, ix in enumerate(chosen):
            X_raw[ptr] = X_full[ix]
            y_raw[ptr] = c
            ptr += 1
        print(f"  Class {c}: Loaded {N_SAMPLES_PER_CLASS} samples")
    del X_full, y_full
    gc.collect()
    print(f"  Subsampled: {X_raw.shape}")

    # Downsample to symbol rate
    I_full = X_raw[:, :, 0]
    I_sym = I_full[:, SPS//2::SPS]
    Q_sym = X_raw[:, SPS//2::SPS, 1]
    X_sym = np.stack((I_sym, Q_sym), axis=2)
    print(f"  Symbols: {X_sym.shape}")

    # Extract or load cached features
    cache_path = "sovereign_features_9000.npz"
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path} ...")
        data = np.load(cache_path)
        X_hos = data['X_hos']
        X_syn = data['X_syn']
        X_bit = data['X_bit']
        y_raw = data['y_raw']
    else:
        print(f"Extracting features for {total} samples on {cpu_count()} cores ...")
        X_hos = np.zeros((total, 17), dtype=np.float32)
        X_syn = np.zeros((total, 4), dtype=np.float32)
        X_bit = np.zeros((total, *FEC_MATRIX_SHAPE, 1), dtype=np.float32)

        n_workers = cpu_count()
        chunk_size = max(1, (total + n_workers * 2 - 1) // (n_workers * 2))
        ranges = [(i, min(i + chunk_size, total)) for i in range(0, total, chunk_size)]

        def process_chunk(lo, hi):
            hos_chunk = np.zeros((hi - lo, 17), dtype=np.float32)
            syn_chunk = np.zeros((hi - lo, 4), dtype=np.float32)
            bit_chunk = np.zeros((hi - lo, *FEC_MATRIX_SHAPE), dtype=np.float32)
            for j in range(hi - lo):
                hos_chunk[j] = extract_hos_features(X_sym[lo + j])
                syn_chunk[j] = extract_syndrome_features(X_sym[lo + j])
                bit_chunk[j] = extract_bit_matrix(X_sym[lo + j])
            return lo, hos_chunk, syn_chunk, bit_chunk

        t0 = time.time()
        done_count = 0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(process_chunk, lo, hi) for lo, hi in ranges]
            for f in as_completed(futures):
                lo, hos_chunk, syn_chunk, bit_chunk = f.result()
                hi = lo + len(hos_chunk)
                X_hos[lo:hi] = hos_chunk
                X_syn[lo:hi] = syn_chunk
                X_bit[lo:hi, ..., 0] = bit_chunk
                done_count += hi - lo
                print(f"  Chunk [{lo}:{hi}] done ({done_count}/{total})")
        t_feat = time.time() - t0
        print(f"  Feature extraction time: {t_feat:.1f}s ({total/t_feat:.1f} samples/sec)")
        print(f"  HOS: {X_hos.shape}, Syndrome: {X_syn.shape}, Bit matrices: {X_bit.shape}")

        print(f"Caching features to {cache_path} ...")
        np.savez_compressed(cache_path, X_hos=X_hos, X_syn=X_syn, X_bit=X_bit, y_raw=y_raw)

    # Combine features
    X_feat = np.concatenate([X_hos, X_syn], axis=1).astype(np.float32)
    print(f"  Combined features: {X_feat.shape}")

    acc = train_eval_sovereign(X_bit, X_feat, y_raw, "Sovereign_RRC_Syndrome_HOS")

    if acc >= 0.80:
        print("\n  VERDICT: RRC + Syndrome Kernel + 2D CNN → FEC identification SUCCESS!")
    elif acc >= 0.40:
        print(f"\n  VERDICT: Significant improvement ({acc*100:.1f}%) — structural features working.")
    elif acc > 0.20:
        print(f"\n  VERDICT: Marginal ({acc*100:.1f}%) — more feature engineering needed.")
    else:
        print(f"\n  VERDICT: RRC alone insufficient with current features.")

    return acc

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SOVEREIGN RECONSTRUCTION — G.975.1 FEC Identification")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"AI_WINDOW={AI_WINDOW}, SPS={SPS}, RRC alpha={dataset_mod.RRC_ALPHA}")

    # Step 1: RRC Eye Diagram verification
    rrc_eye_diagram()
    gc.collect()

    # Step 2: Sovereign Reconstruction Test
    sovereign_test()
    gc.collect()

    print("\n" + "=" * 70)
    print("SOVEREIGN RECONSTRUCTION — COMPLETE")
    print("=" * 70)
