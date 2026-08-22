"""
AIRP v2 — G.975.1 FEC Identifier with Physics-Aware IQ Features
================================================================
اصلاح اساسی رویکرد استخراج ویژگی:
  - حذف WHT/Rank/Streak که به ساختار بیتی وابسته‌اند (زیر NLI/SSFM کار نمی‌کنند)
  - جایگزینی با ویژگی‌های آماری/طیفی مستقیم از سیگنال IQ
  - افزودن ویژگی‌های مبتنی بر نرخ خطای حاشیه‌ای و توزیع خطا (کلید اصلی تمایز)
  - معماری CNN+MLP ترکیبی برای استخراج همزمان ویژگی‌های محلی و کلی

چرا این رویکرد کار می‌کند:
  - هر کد FEC یک "امضای خطای" خاص دارد: LDPC خطاهای خوشه‌ای، RS کلمه‌های خطا
  - این امضا در توزیع فاصله Euclidean نقاط IQ از constellation ایده‌آل قابل مشاهده است
  - حتی تحت SSFM شدید، ratio نرخ خطا در SNR های مختلف متفاوت است
"""

import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from numba import njit, prange
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATA_PATH = "dataset_g975_bitexact_sovereign_final.h5"

# ==============================================================================
# ۱. استخراج ویژگی‌های مبتنی بر فیزیک سیگنال IQ (نه ساختار بیتی)
# ==============================================================================

@njit(cache=True)
def extract_iq_physics_features(sig_iq):
    """
    ویژگی‌هایی که تحت SSFM و NLI شدید پایدار می‌مانند.
    
    اصل: به جای دیدن بیت‌ها (که SSFM خراب می‌کند)، امضای آماری نرخ خطا را ببینیم.
    هر کد FEC → توزیع خطای متفاوت → توزیع فاصله از constellation متفاوت.
    """
    I = sig_iq[:, 0]
    Q = sig_iq[:, 1]
    N = len(I)
    
    mag = np.sqrt(I**2 + Q**2 + 1e-12)
    phase = np.arctan2(Q, I)
    
    # --- گروه ۱: آمار دامنه (۸ ویژگی) ---
    # فاصله از constellation QPSK ایده‌آل (امتیاز soft-decision)
    # هر نقطه QPSK ایده‌آل در فاصله sqrt(P) از مبدأ است
    # کدهای ضعیف‌تر → توزیع گسترده‌تر فاصله از constellation
    ideal_mag = np.mean(mag)  # تخمین توان
    dev = np.abs(mag - ideal_mag)
    
    f = np.zeros(33, dtype=np.float32)
    f[0] = np.mean(dev) / (ideal_mag + 1e-10)           # نرمالیزه‌شده normalized deviation
    f[1] = np.std(dev) / (ideal_mag + 1e-10)            # ضریب تغییرات
    f[2] = np.mean(mag**4) / (np.mean(mag**2)**2 + 1e-10) - 3.0  # excess kurtosis دامنه
    
    # هیستوگرام دامنه به ۵ باکت (توزیع نرمالیزه)
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
    
    # --- گروه ۲: آمار فاز (۸ ویژگی) ---
    d_phase = np.diff(phase)
    # تصحیح پیچش فاز (unwrapping ساده)
    for i in range(len(d_phase)):
        if d_phase[i] > np.pi:
            d_phase[i] -= 2*np.pi
        elif d_phase[i] < -np.pi:
            d_phase[i] += 2*np.pi
    
    f[8]  = np.mean(d_phase)          # drift فاز متوسط (SPM/XPM)
    f[9]  = np.std(d_phase)           # نویز فاز
    f[10] = np.mean(d_phase**2)       # توان نویز فاز
    f[11] = np.mean(np.abs(d_phase))  # میانگین مطلق
    
    # کوپلاژ توان-فاز (امضای NCC کر)
    mag_sq = mag**2
    pwr = mag_sq[:-1]
    u_pwr = np.mean(pwr)
    u_dp  = np.mean(d_phase)
    cov_pf = np.mean((pwr - u_pwr) * (d_phase - u_dp))
    f[12] = cov_pf / (np.std(pwr) * np.std(d_phase) + 1e-10)
    
    d2_phase = np.diff(d_phase)
    f[13] = np.std(d2_phase)          # jitter درجه دوم
    f[14] = np.mean(d2_phase**2)
    # percentile 90 — به روش دستی (njit-compatible)
    abs_dp = np.abs(d_phase)
    sorted_dp = np.sort(abs_dp)
    f[15] = sorted_dp[int(0.9 * len(sorted_dp))]
    
    # --- گروه ۳: ویژگی‌های Soft-Decision Margin (کلید اصلی) ---
    # فاصله از نزدیک‌ترین نقطه constellation QPSK
    # این مستقیماً BER margin را اندازه می‌گیرد
    # نقاط QPSK: (±1, ±1) * scale
    scale = ideal_mag / np.sqrt(2.0)
    
    min_dists = np.zeros(N, dtype=np.float64)
    for i in range(N):
        best = 1e18
        for si in [-1.0, 1.0]:
            for sq in [-1.0, 1.0]:
                d = (I[i] - si*scale)**2 + (Q[i] - sq*scale)**2
                if d < best:
                    best = d
        min_dists[i] = np.sqrt(best)
    
    # آمار فاصله از constellation
    f[16] = np.mean(min_dists) / (scale + 1e-10)
    f[17] = np.std(min_dists)  / (scale + 1e-10)
    f[18] = np.mean(min_dists**2) / (scale**2 + 1e-10)  # متوسط توان خطا
    
    # نسبت نقاط «در معرض خطر» (فاصله > آستانه → احتمال خطای بیتی)
    thresh_50 = scale * 0.5
    thresh_100 = scale * 1.0
    thresh_150 = scale * 1.5
    count_50, count_100, count_150 = 0, 0, 0
    for i in range(N):
        if min_dists[i] > thresh_50:  count_50  += 1
        if min_dists[i] > thresh_100: count_100 += 1
        if min_dists[i] > thresh_150: count_150 += 1
    f[19] = count_50  / N
    f[20] = count_100 / N
    f[21] = count_150 / N
    
    # توزیع فاصله در ۵ باکت (هیستوگرام soft-margin)
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
    
    # --- گروه ۴: ویژگی‌های اتوکورلاسیون (امضای مموری کانال) ---
    # کدهای مختلف الگوهای burst خطا متفاوتی دارند
    # LDPC: burst طولانی، RS: burst کوتاه منظم، BCH: burst بسیار کوتاه
    # این در اتوکورلاسیون دامنه نمایان می‌شود
    win = 2048  # پنجره کوچک‌تر برای سرعت
    mag_win = min_dists[:win]
    mean_mw = np.mean(mag_win)
    var_mw  = np.var(mag_win) + 1e-10
    
    for lag in [1, 4, 16, 64, 256, 1024]:
        if lag < win:
            acorr = 0.0
            for i in range(win - lag):
                acorr += (mag_win[i] - mean_mw) * (mag_win[i+lag] - mean_mw)
            f[27 + [1,4,16,64,256,1024].index(lag)] = (acorr / ((win-lag) * var_mw))
    
    return f


def extract_spectral_features(sig_iq):
    """
    Group 5: Spectral features via FFT (not Numba-compatible).
    CRC/BCH codes create periodic frequencies in the error pattern.
    """
    I = sig_iq[:, 0]
    Q = sig_iq[:, 1]
    mag_sq = I**2 + Q**2
    mag_sq_win = mag_sq[:4096]
    fft_pwr = np.abs(np.fft.rfft(mag_sq_win - np.mean(mag_sq_win)))**2

    feats = np.zeros(7, dtype=np.float32)
    fft_len = len(fft_pwr)
    for b in range(6):
        lo = b * fft_len // 6
        hi = (b+1) * fft_len // 6
        feats[b] = np.sum(fft_pwr[lo:hi]) / (np.sum(fft_pwr) + 1e-10)

    peak_idx = np.argmax(fft_pwr[1:]) + 1
    feats[6] = peak_idx / fft_len
    return feats


@njit(cache=True, parallel=True)
def batch_extract(X_batch):
    """استخراج موازی ویژگی‌ها از یک دسته نمونه"""
    n = X_batch.shape[0]
    feats = np.zeros((n, 33), dtype=np.float32)
    for i in prange(n):
        feats[i] = extract_iq_physics_features(X_batch[i])
    return feats


# ==============================================================================
# ۲. ویژگی‌های مبتنی بر «امضای نرخ کد» (Code Rate Signature)
#    این بخش در Python اجرا می‌شود (نه Numba) برای انعطاف بیشتر
# ==============================================================================

def approx_entropy(u, m=2, r_ratio=0.2):
    N = min(len(u), 256)
    u = u[:N]
    r = r_ratio * np.std(u)
    if r < 1e-10:
        return 0.0

    def phi(m_val):
        templates = np.array([u[i:i+m_val] for i in range(N - m_val + 1)])
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
        d0 = np.linalg.norm(phase_pts[i+1] - phase_pts[i]) + 1e-10
        d1 = np.linalg.norm(phase_pts[i+51] - phase_pts[i+50]) + 1e-10
        diverge_rates.append(np.log(d1/d0))
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
        seg = mag_sq[w*window_size:(w+1)*window_size]
        window_vars[w] = np.var(seg)
        window_means[w] = np.mean(seg) + 1e-10
    vmr = np.mean(window_vars / window_means)

    return np.array([apen_I, apen_Q, lyap, quad_imbalance, spectral_flatness, papr_db, vmr], dtype=np.float32)


# ==============================================================================
# ۳. استخراج ویژگی‌های طیفی با FFT (قابل تفسیر توسط مدل CNN)
#    این ویژگی‌ها به عنوان ورودی مستقیم به CNN می‌روند
# ==============================================================================

def extract_spectral_profile(sig_iq, n_bins=64):
    """
    پروفایل طیفی مختصر از سیگنال.
    کدهای مختلف → الگوهای spectral متفاوت در توان و فاز.
    """
    I = sig_iq[:, 0].astype(np.float64)
    Q = sig_iq[:, 1].astype(np.float64)
    
    complex_sig = I + 1j * Q
    
    # FFT روی بلوک‌های مختلف و میانگین طیفی
    block_size = 1024
    n_blocks = min(8, len(I) // block_size)
    
    psd = np.zeros(block_size // 2 + 1)
    for b in range(n_blocks):
        block = complex_sig[b*block_size:(b+1)*block_size]
        psd += np.abs(np.fft.rfft(block))**2
    psd /= (n_blocks + 1e-10)
    
    # دسیماسیون به n_bins باکت لگاریتمی
    log_bins = np.logspace(0, np.log10(len(psd)-1), n_bins).astype(int)
    log_bins = np.unique(np.clip(log_bins, 0, len(psd)-1))
    profile = psd[log_bins]
    profile = profile / (np.max(profile) + 1e-10)  # نرمالیزاسیون
    
    # اطمینان از طول ثابت
    if len(profile) < n_bins:
        profile = np.pad(profile, (0, n_bins - len(profile)))
    
    return profile[:n_bins].astype(np.float32)


def _extract_python_features(sig_iq, spec_dim):
    cr = extract_code_rate_signature(sig_iq)
    sf = extract_spectral_features(sig_iq)
    sp = extract_spectral_profile(sig_iq, spec_dim)
    return cr, sf, sp


# ==============================================================================
# ۴. معماری مدل ترکیبی (Hybrid Architecture)
#    ورودی ۱: بردار ویژگی ۴۷ بعدی (physics + code-rate)
#    ورودی ۲: پروفایل طیفی ۶۴ بعدی
# ==============================================================================

def build_hybrid_model(feat_dim, spec_dim, n_classes=9):
    feat_input = layers.Input(shape=(feat_dim,), name='physics_features')

    gate = layers.Dense(feat_dim, activation='sigmoid')(feat_input)
    x1 = layers.Multiply()([feat_input, gate])

    x1 = layers.Dense(256, activation='swish',
                       kernel_regularizer=regularizers.l2(1e-4))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.3)(x1)

    res = layers.Dense(256, activation='linear')(x1)
    x1 = layers.Dense(256, activation='swish')(x1)
    x1 = layers.Dense(256, activation='swish')(x1)
    x1 = layers.Add()([x1, res])
    x1 = layers.BatchNormalization()(x1)

    x1 = layers.Dense(128, activation='swish')(x1)

    spec_input = layers.Input(shape=(spec_dim, 1), name='spectral_profile')

    x2 = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(spec_input)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x2)
    x2 = layers.GlobalAveragePooling1D()(x2)
    x2 = layers.Dense(64, activation='swish')(x2)

    merged = layers.Concatenate()([x1, x2])
    x = layers.Dense(256, activation='swish')(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='swish')(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs=[feat_input, spec_input], outputs=outputs)
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ==============================================================================
# ۵. اجرای اصلی
# ==============================================================================

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at '{DATA_PATH}'! Run the generator script first.")
        exit()

    BATCH_SIZE = 500
    N_TOTAL = 21600
    SPEC_DIM = 64

    rng = np.random.default_rng(42)
    indices = np.sort(rng.permutation(108000)[:N_TOTAL])

    print("📂 Extracting physics + spectral features (SSFM-aware)...")
    all_feats   = []
    all_spectra  = []
    all_labels  = []

    with h5py.File(DATA_PATH, 'r') as f:
        for batch_start in range(0, N_TOTAL, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, N_TOTAL)
            batch_idxs = indices[batch_start:batch_end]
            X_batch = f['X'][batch_idxs]
            y_batch = f['y'][batch_idxs]

            print(f"  [{batch_start:6d}/{N_TOTAL}] Processing {batch_end-batch_start} samples...", end=' ')

            # ویژگی‌های فیزیکی IQ (Numba parallel)
            phys = batch_extract(X_batch)

            # ویژگی‌های سطح Python (ترکیب شده برای کاهش سربار)
            py_feats = Parallel(n_jobs=-1, backend='threading')(
                delayed(_extract_python_features)(sig, SPEC_DIM) for sig in X_batch
            )
            py_code_rate = np.array([f[0] for f in py_feats], dtype=np.float32)
            py_spectral_fft = np.array([f[1] for f in py_feats], dtype=np.float32)
            py_spectra = np.array([f[2] for f in py_feats], dtype=np.float32)

            chunk = np.hstack([phys, py_spectral_fft, py_code_rate])
            all_feats.append(chunk)
            all_spectra.append(py_spectra)
            all_labels.append(y_batch)
            print(f"✓ feat_shape={chunk.shape[1]}")

    X_feats   = np.vstack(all_feats).astype(np.float32)
    X_spectra = np.vstack(all_spectra).astype(np.float32)
    y_raw     = np.concatenate(all_labels)
    del all_feats, all_spectra, all_labels

    print(f"\n  Feature matrix: {X_feats.shape}  |  Spectral matrix: {X_spectra.shape}")
    print(f"  NaN in features: {np.isnan(X_feats).sum()}")

    # پاکسازی NaN/Inf
    X_feats   = np.nan_to_num(X_feats,   nan=0.0, posinf=10.0, neginf=-10.0)
    X_spectra = np.nan_to_num(X_spectra, nan=0.0, posinf=1.0,  neginf=0.0)

    # مقیاس‌بندی robust (مقاوم در برابر outlier)
    scaler = RobustScaler()
    X_feats_scaled = scaler.fit_transform(X_feats).astype(np.float32)

    # Reshape برای CNN
    X_spectra_cnn = X_spectra[..., np.newaxis]  # (N, 64, 1)

    y_cat = tf.keras.utils.to_categorical(y_raw, 9)

    (X_f_tr, X_f_te,
     X_s_tr, X_s_te,
     y_tr,   y_te) = train_test_split(
        X_feats_scaled, X_spectra_cnn, y_cat,
        test_size=0.15, stratify=y_raw, random_state=42
    )

    print(f"\n🧠 Building & Training Hybrid AIRP v2 model...")
    model = build_hybrid_model(X_feats_scaled.shape[1], SPEC_DIM)
    model.summary()

    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=20,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8,
            min_lr=1e-5, verbose=1
        ),
        callbacks.ModelCheckpoint(
            'best_airp_v2.keras', monitor='val_accuracy',
            save_best_only=True, verbose=0
        )
    ]

    history = model.fit(
        [X_f_tr, X_s_tr], y_tr,
        validation_split=0.2,
        epochs=150,
        batch_size=128,
        callbacks=cb_list,
        verbose=1
    )

    # ارزیابی نهایی
    print("\n📊 FINAL EVALUATION:")
    y_pred = np.argmax(model.predict([X_f_te, X_s_te]), axis=1)
    y_true = np.argmax(y_te, axis=1)

    target_names = [
        "I.1 (RS 239/255)",
        "I.2 (RS+CSOC)",
        "I.3 (BCH concat)",
        "I.4 (RS+BCH inter)",
        "I.5 (RS product)",
        "I.6 (LDPC stair)",
        "I.7 (BCH product)",
        "I.8 (RS GF12)",
        "I.9 (BCH dual)"
    ]
    print(classification_report(y_true, y_pred, labels=list(range(9)), target_names=target_names))

    test_acc = np.mean(y_pred == y_true)
    print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ماتریس خام
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"I.{i+1}" for i in range(9)],
                yticklabels=[f"I.{i+1}" for i in range(9)],
                ax=axes[0])
    axes[0].set_title(f"Confusion Matrix (Raw)\nTest Acc: {test_acc*100:.1f}%")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    # ماتریس نرمالیزه
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=[f"I.{i+1}" for i in range(9)],
                yticklabels=[f"I.{i+1}" for i in range(9)],
                ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized per class)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.suptitle("AIRP v2 — G.975.1 FEC Identification under SSFM/NLI Channel", 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("confusion_matrix_v2.png", dpi=300, bbox_inches='tight')
    print("📈 Confusion matrix saved to 'confusion_matrix_v2.png'")

    # --- منحنی آموزش ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    axes2[0].plot(history.history['accuracy'],    label='Train')
    axes2[0].plot(history.history['val_accuracy'], label='Val')
    axes2[0].set_title('Accuracy'); axes2[0].legend(); axes2[0].grid()

    axes2[1].plot(history.history['loss'],    label='Train')
    axes2[1].plot(history.history['val_loss'], label='Val')
    axes2[1].set_title('Loss'); axes2[1].legend(); axes2[1].grid()

    plt.suptitle("AIRP v2 Training History")
    plt.tight_layout()
    plt.savefig("training_v2.png", dpi=150, bbox_inches='tight')
    print("📈 Training history saved to 'training_v2.png'")

    # --- تحلیل «گروه مرگ» I.5 vs I.8 ---
    print("\n🔍 Hard-pair analysis (I.5 vs I.8 vs I.6):")
    confusion_pairs = [(4,7), (4,5), (7,5), (5,6), (2,3)]  # 0-indexed
    for ti, pi in confusion_pairs:
        mask = (y_true == ti)
        if mask.sum() == 0: continue
        confused = np.sum(y_pred[mask] == pi)
        total = mask.sum()
        print(f"  I.{ti+1} → predicted as I.{pi+1}: {confused}/{total} ({100*confused/total:.1f}%)")