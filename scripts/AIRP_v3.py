import os, warnings, gc, pickle, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/tmp/cuda_xla'
warnings.filterwarnings('ignore')

import h5py
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit, prange
from joblib import Parallel, delayed

DATA_PATH = "dataset_g975_bitexact_sovereign_final.h5"
FEAT_CACHE = "features_v3_cache.npy"

N_WINDOWS  = 8
BATCH_SIZE = 32
N_PER_CLASS = 1200
EPOCHS = 100
N_CLASSES = 9
WINDOW_SIZE = 16384 // N_WINDOWS
FEAT_DIM = 47

rng = np.random.default_rng(42)
np.random.seed(42)
tf.random.set_seed(42)


@njit(cache=True)
def extract_iq_physics_features(sig_iq):
    I = sig_iq[:, 0]
    Q = sig_iq[:, 1]
    N = len(I)

    mag = np.sqrt(I**2 + Q**2 + 1e-12)
    phase = np.arctan2(Q, I)

    ideal_mag = np.mean(mag)
    dev = np.abs(mag - ideal_mag)

    f = np.zeros(33, dtype=np.float32)
    f[0] = np.mean(dev) / (ideal_mag + 1e-10)
    f[1] = np.std(dev) / (ideal_mag + 1e-10)
    f[2] = np.mean(mag**4) / (np.mean(mag**2)**2 + 1e-10) - 3.0

    hist_min = ideal_mag * 0.0
    hist_max = ideal_mag * 2.5
    bin_w = (hist_max - hist_min) / 5.0
    for b in range(5):
        count = 0
        lo = hist_min + b * bin_w
        hi = lo + bin_w
        for i in range(N):
            if mag[i] >= lo and mag[i] < hi:
                count += 1
        f[3 + b] = count / N

    d_phase = np.diff(phase)
    for i in range(len(d_phase)):
        if d_phase[i] > np.pi:
            d_phase[i] -= 2 * np.pi
        elif d_phase[i] < -np.pi:
            d_phase[i] += 2 * np.pi

    f[8] = np.mean(d_phase)
    f[9] = np.std(d_phase)
    f[10] = np.mean(d_phase**2)
    f[11] = np.mean(np.abs(d_phase))

    mag_sq = mag**2
    pwr = mag_sq[:-1]
    u_pwr = np.mean(pwr)
    u_dp = np.mean(d_phase)
    cov_pf = np.mean((pwr - u_pwr) * (d_phase - u_dp))
    f[12] = cov_pf / (np.std(pwr) * np.std(d_phase) + 1e-10)

    d2_phase = np.diff(d_phase)
    f[13] = np.std(d2_phase)
    f[14] = np.mean(d2_phase**2)
    abs_dp = np.abs(d_phase)
    sorted_dp = np.sort(abs_dp)
    f[15] = sorted_dp[int(0.9 * len(sorted_dp))]

    scale = ideal_mag / np.sqrt(2.0)
    min_dists = np.zeros(N, dtype=np.float64)
    for i in range(N):
        best = 1e18
        for si in [-1.0, 1.0]:
            for sq in [-1.0, 1.0]:
                d = (I[i] - si * scale)**2 + (Q[i] - sq * scale)**2
                if d < best:
                    best = d
        min_dists[i] = np.sqrt(best)

    f[16] = np.mean(min_dists) / (scale + 1e-10)
    f[17] = np.std(min_dists) / (scale + 1e-10)
    f[18] = np.mean(min_dists**2) / (scale**2 + 1e-10)

    thresh_50 = scale * 0.5
    thresh_100 = scale * 1.0
    thresh_150 = scale * 1.5
    count_50, count_100, count_150 = 0, 0, 0
    for i in range(N):
        if min_dists[i] > thresh_50:
            count_50 += 1
        if min_dists[i] > thresh_100:
            count_100 += 1
        if min_dists[i] > thresh_150:
            count_150 += 1
    f[19] = count_50 / N
    f[20] = count_100 / N
    f[21] = count_150 / N

    max_d = scale * 2.0
    bin_w2 = max_d / 5.0
    for b in range(5):
        count = 0
        lo = b * bin_w2
        hi = lo + bin_w2
        for i in range(N):
            if min_dists[i] >= lo and min_dists[i] < hi:
                count += 1
        f[22 + b] = count / N

    win = 2048
    mag_win = min_dists[:win]
    mean_mw = np.mean(mag_win)
    var_mw = np.var(mag_win) + 1e-10
    for lag_idx, lag in enumerate([1, 4, 16, 64, 256, 1024]):
        if lag < win:
            acorr = 0.0
            for i in range(win - lag):
                acorr += (mag_win[i] - mean_mw) * (mag_win[i + lag] - mean_mw)
            f[27 + lag_idx] = (acorr / ((win - lag) * var_mw))

    return f


@njit(cache=True, parallel=True)
def batch_extract_physics(X_batch):
    n = X_batch.shape[0]
    feats = np.zeros((n, 33), dtype=np.float32)
    for i in prange(n):
        feats[i] = extract_iq_physics_features(X_batch[i])
    return feats


def approx_entropy(u, m=2, r_ratio=0.2):
    N = min(len(u), 256)
    u = u[:N]
    r = r_ratio * np.std(u)
    if r < 1e-10:
        return 0.0

    def phi(m_val):
        templates = np.array([u[i:i + m_val] for i in range(N - m_val + 1)])
        count = np.zeros(len(templates))
        for i, t in enumerate(templates):
            diff = np.max(np.abs(templates - t), axis=1)
            count[i] = np.sum(diff <= r)
        return np.mean(np.log(count / (N - m_val + 1) + 1e-10))

    return abs(phi(m) - phi(m + 1))


def extract_code_rate_signature(sig_iq):
    I_comp = sig_iq[:, 0]
    Q_comp = sig_iq[:, 1]

    apen_I = approx_entropy(I_comp)
    apen_Q = approx_entropy(Q_comp)

    phase_pts = np.column_stack([I_comp[:512], Q_comp[:512]])
    diverge_rates = []
    for i in range(0, 500, 50):
        d0 = np.linalg.norm(phase_pts[i + 1] - phase_pts[i]) + 1e-10
        d1 = np.linalg.norm(phase_pts[i + 51] - phase_pts[i + 50]) + 1e-10
        diverge_rates.append(np.log(d1 / d0))
    lyap = np.mean(diverge_rates)

    q1 = np.sum((I_comp > 0) & (Q_comp > 0)) / len(I_comp)
    q2 = np.sum((I_comp < 0) & (Q_comp > 0)) / len(I_comp)
    q3 = np.sum((I_comp < 0) & (Q_comp < 0)) / len(I_comp)
    q4 = np.sum((I_comp > 0) & (Q_comp < 0)) / len(I_comp)
    quad_imbalance = np.std([q1, q2, q3, q4])

    mag_sq = I_comp**2 + Q_comp**2
    fft_mag = np.abs(np.fft.rfft(mag_sq[:2048]))**2
    fft_mag = fft_mag[1:] + 1e-10
    geo_mean = np.exp(np.mean(np.log(fft_mag)))
    ari_mean = np.mean(fft_mag)
    spectral_flatness = geo_mean / (ari_mean + 1e-10)

    papr = np.max(mag_sq) / (np.mean(mag_sq) + 1e-10)
    papr_db = 10 * np.log10(papr + 1e-10)

    window_size = 256
    n_windows = len(I_comp) // window_size
    window_vars = np.zeros(n_windows)
    window_means = np.zeros(n_windows)
    for w in range(n_windows):
        seg = mag_sq[w * window_size:(w + 1) * window_size]
        window_vars[w] = np.var(seg)
        window_means[w] = np.mean(seg) + 1e-10
    vmr = np.mean(window_vars / window_means)

    return np.array([apen_I, apen_Q, lyap, quad_imbalance, spectral_flatness, papr_db, vmr], dtype=np.float32)


def extract_spectral_features(sig_iq):
    I = sig_iq[:, 0]
    Q = sig_iq[:, 1]
    mag_sq = I**2 + Q**2
    mag_sq_win = mag_sq[:4096]
    fft_pwr = np.abs(np.fft.rfft(mag_sq_win - np.mean(mag_sq_win)))**2

    feats = np.zeros(7, dtype=np.float32)
    fft_len = len(fft_pwr)
    for b in range(6):
        lo = b * fft_len // 6
        hi = (b + 1) * fft_len // 6
        feats[b] = np.sum(fft_pwr[lo:hi]) / (np.sum(fft_pwr) + 1e-10)

    peak_idx = np.argmax(fft_pwr[1:]) + 1
    feats[6] = peak_idx / fft_len
    return feats


def extract_all_features(sig_iq):
    physics = extract_iq_physics_features(sig_iq)
    cr = extract_code_rate_signature(sig_iq)
    sf = extract_spectral_features(sig_iq)
    return np.concatenate([physics, cr, sf]).astype(np.float32)


def precompute_features(indices, cache_path=FEAT_CACHE):
    phys_path = cache_path.replace('.npy', '_phys.npy')
    other_path = cache_path.replace('.npy', '_other.npy')

    if os.path.exists(phys_path) and os.path.exists(other_path):
        print(f"    Loading cached features from {phys_path}, {other_path}")
        phys_feats = np.load(phys_path)
        other_feats = np.load(other_path)
        return np.concatenate([phys_feats, other_feats], axis=1)

    print(f"    Extracting physics features from {len(indices)} samples (Numba parallel)...")
    t0 = time.time()
    with h5py.File(DATA_PATH, 'r') as f:
        X_all = f['X'][indices].astype(np.float32)
    phys_feats = batch_extract_physics(X_all)
    print(f"    Physics done in {time.time() - t0:.1f}s")

    print(f"    Extracting code-rate + spectral features (joblib)...")
    t0 = time.time()
    other_feats = Parallel(n_jobs=-1, verbose=0)(
        delayed(lambda i: np.concatenate([
            extract_code_rate_signature(X_all[i]),
            extract_spectral_features(X_all[i])
        ]).astype(np.float32))(i) for i in range(len(X_all))
    )
    other_feats = np.array(other_feats, dtype=np.float32)
    print(f"    Code-rate+spectral done in {time.time() - t0:.1f}s")

    np.save(phys_path, phys_feats)
    np.save(other_path, other_feats)

    feats = np.concatenate([phys_feats, other_feats], axis=1)
    return feats


def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, gamma)
        if alpha is not None:
            alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
            weight = weight * alpha_t
        return tf.reduce_sum(weight * ce, axis=-1)
    return loss_fn


def residual_block(x, filters, kernel_size=3, stride=1, dilation=1):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same',
                      dilation_rate=dilation, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same',
                      dilation_rate=dilation, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if stride > 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def build_hybrid_model(window_size, feat_dim, n_classes=9):
    iq_input = layers.Input(shape=(window_size, 2), name='raw_iq')
    x = layers.Conv1D(32, 7, strides=2, padding='same', use_bias=False)(iq_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    x = residual_block(x, 64,  kernel_size=3, stride=2, dilation=1)
    x = residual_block(x, 128, kernel_size=3, stride=1, dilation=2)
    x = residual_block(x, 256, kernel_size=3, stride=2, dilation=1)
    x = residual_block(x, 256, kernel_size=3, stride=1, dilation=4)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    feat_input = layers.Input(shape=(feat_dim,), name='physics_features')
    f = layers.Dense(128, activation='relu')(feat_input)
    f = layers.BatchNormalization()(f)
    f = layers.Dense(128, activation='relu')(f)
    f = layers.BatchNormalization()(f)

    merged = layers.Concatenate()([x, f])
    x = layers.Dense(256, activation='relu')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs=[iq_input, feat_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    return model


class HybridSequence(tf.keras.utils.Sequence):
    def __init__(self, indices, features, batch_size, n_windows, phase_aug=True, shuffle=True):
        self.indices = indices.copy().astype(np.int64)
        self.features = features.astype(np.float32)
        self.batch_size = batch_size
        self.n_windows = n_windows
        self.window_size = 16384 // n_windows
        self.phase_aug = phase_aug
        self.shuffle = shuffle
        self.h5 = h5py.File(DATA_PATH, 'r')
        self.idx_to_feat = {int(raw): int(pos) for pos, raw in enumerate(indices)}
        self._verified = False
        if self.shuffle:
            np.random.shuffle(self.indices)
        self._verify_indices()

    def _verify_indices(self):
        check = self.indices[:min(10, len(self.indices))]
        for raw in check:
            pos = self.idx_to_feat[int(raw)]
            if pos < 0 or pos >= len(self.features):
                raise ValueError(f"Index {raw} maps to feature pos {pos} (feat len={len(self.features)})")
            y_h5 = int(self.h5['y'][int(raw)])
            if not (0 <= y_h5 < 9):
                raise ValueError(f"Label {y_h5} out of range at HDF5 index {raw}")
        print(f"  [VERIFIED] {len(check)} samples: indices, features, labels aligned")

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_idx_sorted = np.sort(batch_idx)

        if self.phase_aug:
            w = np.random.randint(0, self.n_windows)
        else:
            w = 0
        start = w * self.window_size
        X_iq = self.h5['X'][batch_idx_sorted, start:start + self.window_size, :].astype(np.float32)
        y = self.h5['y'][batch_idx_sorted]

        if self.phase_aug:
            X_iq = self._rotate_batch(X_iq)

        feat_pos = [self.idx_to_feat[int(idx)] for idx in batch_idx_sorted]
        X_feat = self.features[feat_pos]
        y_cat = tf.keras.utils.to_categorical(y, N_CLASSES)
        return (X_iq, X_feat), y_cat

    def _rotate_batch(self, X):
        thetas = np.random.uniform(-np.pi, np.pi, X.shape[0])
        for i in range(X.shape[0]):
            c, s = np.cos(thetas[i]), np.sin(thetas[i])
            Ii = X[i, :, 0].copy()
            Qi = X[i, :, 1].copy()
            X[i, :, 0] = Ii * c - Qi * s
            X[i, :, 1] = Ii * s + Qi * c
        return X

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __del__(self):
        if hasattr(self, 'h5'):
            self.h5.close()


print("=" * 65)
print("AIRP v3 — Hybrid Mastermind: Raw IQ (CNN ResNet-1D) + Features (MLP)")
print(f"  N_WINDOWS={N_WINDOWS}  WINDOW_SIZE={WINDOW_SIZE}  BATCH_SIZE={BATCH_SIZE}")
print(f"  N_PER_CLASS={N_PER_CLASS}  FEAT_DIM={FEAT_DIM}")
print("=" * 65)

print("\nLoading labels to build stratified indices...")
with h5py.File(DATA_PATH, 'r') as f:
    all_y = f['y'][:]

idx_by_class = {c: np.where(all_y == c)[0] for c in range(N_CLASSES)}
train_idxs, test_idxs = [], []
for c in range(N_CLASSES):
    perm = rng.permutation(idx_by_class[c])
    selected = perm[:N_PER_CLASS]
    tr, te = train_test_split(selected, test_size=0.15, random_state=c)
    train_idxs.append(tr)
    test_idxs.append(te)

train_indices = np.concatenate(train_idxs).astype(np.int64)
test_indices = np.concatenate(test_idxs).astype(np.int64)
rng.shuffle(train_indices)
rng.shuffle(test_indices)

all_indices = np.sort(np.concatenate([train_indices, test_indices]))
idx_to_pos = {idx: pos for pos, idx in enumerate(all_indices)}
print(f"  Train: {len(train_indices)}   Test: {len(test_indices)}   Total: {len(all_indices)}")

overlap = set(train_indices) & set(test_indices)
if overlap:
    raise ValueError(f"TRAIN/TEST OVERLAP: {len(overlap)} indices in both sets!")
print("  Train/test split: no overlap verified")

print("\nPrecomputing features for all selected samples...")
all_features = precompute_features(all_indices)
print(f"  Features shape: {all_features.shape}")

assert not np.any(np.isnan(all_features)), "NaN in features!"
assert not np.any(np.isinf(all_features)), "Inf in features!"
print("  Features: no NaN/Inf verified")

train_features = np.array([all_features[idx_to_pos[i]] for i in train_indices], dtype=np.float32)
test_features = np.array([all_features[idx_to_pos[i]] for i in test_indices], dtype=np.float32)

print("\nScaling features with StandardScaler (fit on train only)...")
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features).astype(np.float32)
test_features = scaler.transform(test_features).astype(np.float32)

print("\nBuilding hybrid dual-stream model (ResNet-1D + MLP)...")
model = build_hybrid_model(WINDOW_SIZE, FEAT_DIM, N_CLASSES)
model.summary(line_length=90)

train_seq = HybridSequence(train_indices, train_features, BATCH_SIZE, N_WINDOWS, phase_aug=True, shuffle=True)
val_seq = HybridSequence(test_indices, test_features, BATCH_SIZE, N_WINDOWS, phase_aug=False, shuffle=False)

cb_list = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=5e-6, verbose=1),
    callbacks.ModelCheckpoint('best_airp_v3.h5', monitor='val_accuracy', save_best_only=True, verbose=0),
]

print("\nTraining...")
history = model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=EPOCHS,
    callbacks=cb_list,
    verbose=1
)

del train_seq
gc.collect()

print("\nEvaluating on test set (multi-window voting)...")
N_WIN_EVAL = N_WINDOWS
all_window_preds = []
sorted_test = np.sort(test_indices)
test_idx_to_feat = {int(idx): int(pos) for pos, idx in enumerate(test_indices)}
with h5py.File(DATA_PATH, 'r') as f:
    for w in range(N_WIN_EVAL):
        start = w * (16384 // N_WIN_EVAL)
        end = start + (16384 // N_WIN_EVAL)
        win_preds = []
        for i in range(0, len(sorted_test), BATCH_SIZE * 2):
            batch = sorted_test[i:i + BATCH_SIZE * 2]
            X_iq = f['X'][batch, start:end, :].astype(np.float32)
            feat_idx = [test_idx_to_feat[int(idx)] for idx in batch]
            X_feat = test_features[feat_idx]
            if len(batch) == 0:
                continue
            win_preds.append(model.predict([X_iq, X_feat], verbose=0))
        all_window_preds.append(np.concatenate(win_preds, axis=0))

y_prob = np.mean(all_window_preds, axis=0)
y_pred = np.argmax(y_prob, axis=1)

with h5py.File(DATA_PATH, 'r') as f:
    y_true = f['y'][sorted_test]

test_acc = np.mean(y_pred == y_true)
print(f"\nTest Accuracy (multi-window vote): {test_acc*100:.2f}%")

target_names = [
    "I.1 (RS 239/255)", "I.2 (RS+CSOC)", "I.3 (BCH concat)",
    "I.4 (RS+BCH inter)", "I.5 (RS product)", "I.6 (LDPC stair)",
    "I.7 (BCH product)", "I.8 (RS GF12)", "I.9 (BCH dual)"
]
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_true, y_pred, labels=list(range(9)), target_names=target_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f"I.{i+1}" for i in range(9)],
            yticklabels=[f"I.{i+1}" for i in range(9)],
            ax=axes[0])
axes[0].set_title(f"Confusion Matrix (Raw)\nTest Acc: {test_acc*100:.1f}%")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=[f"I.{i+1}" for i in range(9)],
            yticklabels=[f"I.{i+1}" for i in range(9)],
            ax=axes[1])
axes[1].set_title("Confusion Matrix (Normalized per class)")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

plt.suptitle("AIRP v3 — Hybrid Mastermind (Raw IQ CNN + Features MLP)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("confusion_matrix_v3.png", dpi=300, bbox_inches='tight')
print("\nconfusion_matrix_v3.png saved")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
axes2[0].plot(history.history['accuracy'], label='Train')
axes2[0].plot(history.history['val_accuracy'], label='Val')
axes2[0].set_title('Accuracy'); axes2[0].legend(); axes2[0].grid()

axes2[1].plot(history.history['loss'], label='Train')
axes2[1].plot(history.history['val_loss'], label='Val')
axes2[1].set_title('Loss'); axes2[1].legend(); axes2[1].grid()
plt.suptitle("AIRP v3 Training History")
plt.tight_layout()
plt.savefig("training_v3.png", dpi=150, bbox_inches='tight')
print("training_v3.png saved")

del val_seq
gc.collect()
print("\nDone.")
