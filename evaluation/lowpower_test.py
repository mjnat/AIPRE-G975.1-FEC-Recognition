#!/usr/bin/env python3
"""Quick low-power test: P_LAUNCH_W = -3dBm (0.5mW), 200 samples/class, 32dB OSNR"""
import os, warnings, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import dataset_g975_1_fixed as dataset_mod
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

rng = np.random.default_rng(42)
PSD_DIM = 1024; N_CLASSES = 9; N_SAMP_CLASS = 100

# Lower launch power
orig_power = dataset_mod.P_LAUNCH_W
dataset_mod.P_LAUNCH_W = 0.5e-3  # -3dBm
print(f"Power: {orig_power*1000:.1f}mW (3dBm) → {dataset_mod.P_LAUNCH_W*1000:.1f}mW (-3dBm)")

g_i3_o, g_i3_i, g_i4_i, g_i7_r, g_i7_c = dataset_mod.get_bch_standard_polynomials()
gH_i9, gS_i9 = dataset_mod.get_i9_polynomials_exact()
T_i9 = dataset_mod.get_i9_superposition_matrix_stable(gH_i9, gS_i9)
G_mat = dataset_mod.get_ldpc_g_matrix()

total = N_CLASSES * N_SAMP_CLASS
X = np.zeros((total, 16384, 2), dtype=np.float32)
y = np.zeros(total, dtype=np.int32)

ptr = 0
for c in range(N_CLASSES):
    for i in range(N_SAMP_CLASS):
        X[ptr] = dataset_mod.generate_standard_sovereign_frame(c, 32, g_i3_o, g_i3_i, g_i4_i, g_i7_r, g_i7_c, G_mat, T_i9)
        y[ptr] = c
        ptr += 1
    print(f"  Class {c}: Done")
dataset_mod.P_LAUNCH_W = orig_power

# PSD
print("Extracting PSD...")
def psd(iq):
    I, Q = iq[:,0].astype(np.float64), iq[:,1].astype(np.float64)
    cs = I + 1j*Q; nf=2048; nw=len(cs)//nf; w=np.hanning(nf)
    ps=np.zeros(nf//2+1); seg=np.zeros(nf, dtype=np.complex128)
    for wi in range(nw):
        seg[:]=cs[wi*nf:(wi+1)*nf]; seg*=w
        ff=np.fft.fft(seg); ps+=np.real(ff[:nf//2+1])**2+np.imag(ff[:nf//2+1])**2
    ps/=nw; psd=10*np.log10(np.maximum(ps,1e-15))
    return np.interp(np.linspace(0,len(psd)-1,PSD_DIM),np.arange(len(psd)),psd).astype(np.float32)

X_psd = np.array([psd(X[i]) for i in range(total)])
print(f"PSD: {X_psd.shape}")

# Split
ixc = {c:np.where(y==c)[0] for c in range(N_CLASSES)}
tr, te = [], []
for c in range(N_CLASSES):
    p=rng.permutation(ixc[c]); a,b=train_test_split(p,test_size=0.2,random_state=c)
    tr.append(a); te.append(b)
tr=np.sort(np.concatenate(tr)); te=np.sort(np.concatenate(te))
Xt, Xv = X_psd[tr], X_psd[te]; yt, yv = y[tr], y[te]
print(f"Train: {len(tr)} Test: {len(te)}")

sc=StandardScaler(); Xt=sc.fit_transform(Xt).astype(np.float32); Xv=sc.transform(Xv).astype(np.float32)
Xt=Xt[...,np.newaxis]; Xv=Xv[...,np.newaxis]
ytc=tf.keras.utils.to_categorical(yt,9); yvc=tf.keras.utils.to_categorical(yv,9)

inp=layers.Input(shape=(PSD_DIM,1))
x=layers.Conv1D(32,5,padding='same',use_bias=False)(inp)
x=layers.BatchNormalization()(x); x=layers.Activation('relu')(x); x=layers.MaxPooling1D(4)(x)
x=layers.Conv1D(64,5,padding='same',use_bias=False)(x)
x=layers.BatchNormalization()(x); x=layers.Activation('relu')(x); x=layers.MaxPooling1D(4)(x)
x=layers.Conv1D(128,3,padding='same',use_bias=False)(x)
x=layers.BatchNormalization()(x); x=layers.Activation('relu')(x); x=layers.MaxPooling1D(4)(x)
x=layers.Conv1D(256,3,padding='same',use_bias=False)(x)
x=layers.BatchNormalization()(x); x=layers.Activation('relu')(x); x=layers.GlobalAveragePooling1D()(x)
x=layers.Dense(128,activation='relu',kernel_regularizer=regularizers.l2(1e-4))(x)
x=layers.Dropout(0.4)(x); x=layers.Dense(64,activation='relu')(x)
out=layers.Dense(9,activation='softmax')(x)
m=models.Model(inp,out)
m.compile(optimizer=tf.keras.optimizers.Adam(3e-4),loss='categorical_crossentropy',metrics=['accuracy'])

cb=[callbacks.EarlyStopping(monitor='val_accuracy',patience=20,restore_best_weights=True,verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=10,min_lr=5e-6,verbose=1)]
h=m.fit(Xt,ytc,validation_data=(Xv,yvc),epochs=80,batch_size=32,callbacks=cb,verbose=1)

yp=np.argmax(m.predict(Xv,verbose=0),axis=1); acc=np.mean(yp==yv)
print(f"\n>>> LOW-POWER (-3dBm) 32dB TEST ACCURACY: {acc*100:.2f}% <<<")
tn=["I.1 (RS 239/255)","I.2 (RS+CSOC)","I.3 (BCH concat)","I.4 (RS+BCH inter)","I.5 (RS product)","I.6 (LDPC stair)","I.7 (BCH product)","I.8 (RS GF12)","I.9 (BCH dual)"]
print(classification_report(yv,yp,labels=list(range(9)),target_names=tn,digits=4))

cm=confusion_matrix(yv,yp)
fig,ax=plt.subplots(1,2,figsize=(18,7))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=[f"I.{i+1}" for i in range(9)],yticklabels=[f"I.{i+1}" for i in range(9)],ax=ax[0])
ax[0].set_title(f"Low-Power -3dBm 32dB\nTest Acc: {acc*100:.1f}%"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("True")
cmn=cm.astype(float)/(cm.sum(axis=1,keepdims=True)+1e-10)
sns.heatmap(cmn,annot=True,fmt='.2f',cmap='Greens',xticklabels=[f"I.{i+1}" for i in range(9)],yticklabels=[f"I.{i+1}" for i in range(9)],ax=ax[1])
ax[1].set_title("Normalized"); ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("True")
plt.suptitle("Forensic Rescue — Low-Power (-3dBm) 32dB"); plt.tight_layout()
plt.savefig("confusion_lowpower.png",dpi=300,bbox_inches='tight')
print("confusion_lowpower.png saved")
