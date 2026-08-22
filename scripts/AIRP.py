import os
import h5py
import numpy as np
import tensorflow as tf
import zlib
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import classification_report, confusion_matrix
from numba import njit, prange
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# ۱. تنظیمات موتور شناسایی
# ==============================================================================
DATA_PATH = "dataset_g975_bitexact_sovereign_final.h5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==============================================================================
# ۲. هسته‌های استخراج ویژگی پیشرفته (Numba Optimized)
# ==============================================================================

@njit(cache=True)
def fast_fwht(x):
    """تبدیل والش-هادامارد سریع برای کشف الگوهای XOR در نویز شدید NLI"""
    n = len(x)
    if n == 1: return x
    even = fast_fwht(x[0::2])
    odd = fast_fwht(x[1::2])
    res = np.empty(n, dtype=np.float64)
    half = n // 2
    for i in range(half):
        res[i] = even[i] + odd[i]
        res[i + half] = even[i] - odd[i]
    return res

@njit(cache=True)
def extract_physics_invariants(sig_iq):
    """
    استخراج ناورداهای فیزیکی مقاوم در برابر نویز فاز غیرخطی (NLPN) و پاشندگی (CD)
    """
    I, Q = sig_iq[:, 0], sig_iq[:, 1]
    mag_sq = I**2 + Q**2
    mag = np.sqrt(mag_sq + 1e-10)
    
    # فاز تفاضلی مرتبه اول و دوم (حذف اثر SPM/XPM ثابت)
    phase = np.arctan2(Q, I)
    d_phase = np.diff(phase)
    d2_phase = np.diff(d_phase)
    
    features = np.zeros(18, dtype=np.float32)
    
    # ۱. ویژگی CFV (Cyclic Folding Variance) - شکار رزونانس I.8 در فاصله 2720
    lag = 2720
    num_folds = 5 # با توجه به پنجره 16384
    folded = np.zeros(lag)
    for p in range(num_folds):
        for k in range(lag):
            folded[k] += mag[p*lag + k]
    features[0] = np.var(folded / num_folds)
    
    # ۲. همبستگی غیرخطی (NCC) - جفت‌شدگی توان-فاز (امضای فیزیکی کر)
    # در دیتای شما (SSFM)، این ویژگی کلید تفکیک نویز ASE از NLI است
    u_pwr = np.mean(mag_sq[:-1])
    u_dp = np.mean(d_phase)
    cov = np.mean((mag_sq[:-1] - u_pwr) * (d_phase - u_dp))
    features[1] = cov / (np.std(mag_sq) * np.std(d_phase) + 1e-10)
    
    # ۳. آمارهای مرتبه بالا (HOS) برای تشخیص "بافت" نویز
    features[2] = np.mean(mag**4) / (np.mean(mag**2)**2 + 1e-10) # Kurtosis (دامنه)
    features[3] = np.std(d2_phase) # Jitter نویز فاز غیرخطی
    
    # ۴. امضای جبری WHT روی بیت‌های سخت‌تصمیم (Hard-Decision Bits)
    # اعمال حدآستانه روی IQ برای حذف اثر نویز فاز کر قبل از تبدیل Walsh-Hadamard
    hd_bits = (I > 0).astype(np.int8)
    wht = np.abs(fast_fwht(hd_bits[:1024].astype(np.float64)))
    features[4] = np.max(wht)
    features[5] = np.std(wht)
    features[6] = np.mean(wht)
    
    # ۵. تحلیل فواصل بیت‌های ۱ (Bit-Streak DNA) برای LDPC I.6
    # تشخیص خوشه‌بندی پارتی‌ها در ماتریس پله‌ای
    bits = (I > 0).astype(np.int8)
    streaks = 0
    for i in range(1, 4096):
        if bits[i] == bits[i-1]: streaks += 1
    features[7] = streaks / 4096.0
    
    return features

def extract_morphology_rank(sig_iq):
    """استخراج رتبه ماتریس جبری (کلید اصلی تفکیک I.5 از I.8)"""
    I = sig_iq[:, 0]
    bits = (I > 0).astype(np.uint8)
    
    # ۱. رتبه ماتریس چندمقیاسی (Multi-Scale Matrix Rank)
    # کدهای محصول (I.5) رتبه پایین‌تری نسبت به کدهای خطی (I.8) نشان می‌دهند
    mat64 = bits[:4096].reshape((64, 64)).astype(np.float32)
    rank64 = np.linalg.matrix_rank(mat64) / 64.0
    
    mat128 = bits[:16384].reshape((128, 128)).astype(np.float32)
    rank128 = np.linalg.matrix_rank(mat128) / 128.0
    
    # ۲. نسبت فشرده‌سازی Zlib (آنتروپی جبری)
    z_ratio = len(zlib.compress(np.packbits(bits[:16384]), level=9)) / 2048.0
    
    return [rank64, rank128, z_ratio]

# ==============================================================================
# ۳. معماری هوشمند AIPRE (Gated Residual MLP)
# ==============================================================================

def build_aipre_sovereign_model(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    
    # لایه Gating برای انتخاب هوشمند ویژگی‌ها بر اساس سطح نویز
    gate = layers.Dense(input_dim, activation='sigmoid')(inputs)
    x = layers.Multiply()([inputs, gate])
    
    # بدنه اصلی با اتصالات پسماند (Residual)
    x = layers.Dense(512, activation='swish', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    
    res = x
    x = layers.Dense(512, activation='swish')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation='swish')(x)
    x = layers.Add()([x, res]) # اتصال پسماند برای جلوگیری از محو شدن گرادیان جبر
    
    x = layers.Dense(256, activation='swish')(x)
    outputs = layers.Dense(9, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=5e-4), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==============================================================================
# ۴. اجرای فرآیند شناسایی و ارزیابی
# ==============================================================================

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print("❌ Dataset not found! Run the generator script first.")
        exit()

    BATCH_SIZE = 2000
    N_TOTAL = 108000

    # ۱. استخراج ویژگی‌ها به صورت دسته‌ای (جلوگیری از مصرف ۱۳ گیگابایت رم)
    print("📂 Extracting features in batches from DWDM-NLI Dataset...")
    all_feats = []
    all_labels = []
    with h5py.File(DATA_PATH, 'r') as f:
        for start in range(0, N_TOTAL, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_TOTAL)
            X_batch = f['X'][start:end]
            y_batch = f['y'][start:end]
            
            print(f"  Processing samples {start} to {end}...")
            physics_feats = Parallel(n_jobs=8, backend='threading')(
                delayed(extract_physics_invariants)(sig) for sig in X_batch
            )
            morph_feats = Parallel(n_jobs=8, backend='threading')(
                delayed(extract_morphology_rank)(sig) for sig in X_batch
            )
            
            X_chunk = np.hstack([np.array(physics_feats), np.array(morph_feats)])
            all_feats.append(X_chunk)
            all_labels.append(y_batch)

    X_all = np.vstack(all_feats)
    X_all = np.nan_to_num(X_all)
    y_raw = np.concatenate(all_labels)
    del all_feats, all_labels

    print(f"  Feature matrix shape: {X_all.shape}")

    # ۲. پیش‌پردازش Quantile (حیاتی برای توزیع‌های غیرگوسی نویز کر)
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
    X_scaled = scaler.fit_transform(X_all)
    
    y_cat = tf.keras.utils.to_categorical(y_raw, 9)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.15, stratify=y_raw, random_state=42)

    # ۳. آموزش مدل AIPRE
    print("\n🧠 Training Hierarchical Identifier...")
    model = build_aipre_sovereign_model(X_scaled.shape[1])
    
    # استفاده از EarlyStopping برای جلوگیری از Overfit روی نویز
    cb = [callbacks.EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True)]
    
    model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=64, callbacks=cb)

    # ۴. گزارش نهایی و تحلیل "گروه مرگ"
    print("\n📊 FINAL CLASSIFICATION REPORT (NLI/DWDM Scenario):")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print(classification_report(y_true, y_pred, target_names=[f"I.{i+1}" for i in range(9)]))

    # رسم ماتریس اغتشاش برای اثبات حل تداخل I.5/I.8
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
    plt.title("AIPRE Robust Identification (Severe NLI $P=3$)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.savefig("confusion_matrix_nli.png", dpi=300)
    print("✅ Results saved to 'confusion_matrix_nli.png'")