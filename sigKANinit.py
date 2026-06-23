# train_autoencoder.py
import os, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sigkan import SigKAN          # official package

# ---------- CONFIG ----------
T = 2048
N_CHANNELS = 4
LATENT_DIM = 32                    # bottleneck size
BATCH_SIZE = 64
EPOCHS = 150
FOLD_ID = 0                        # use train groups as train, val groups as val
SIGNAL_DIR = "data/phase2_signals"
METADATA_DIR = "results/phase3"
MODEL_DIR = "models/autoencoder"
# ----------------------------

# ---- 1. Load metadata and folds ----
table = pd.read_csv(os.path.join(METADATA_DIR, "phase3_dataset.csv"))
table['parent_group'] = table['parent_group'].astype(str)
with open(os.path.join(METADATA_DIR, "phase3_folds.json"), "r") as f:
    folds_json = json.load(f)

group_to_fold = folds_json['group_to_fold']
dev_groups = set(map(str, folds_json.get('dev_groups', [])))

# ---- 2. Signal loader ----
def load_signal(file_path, target_length=T):
    data = np.load(file_path)
    signals = np.stack([data['xtpos'], data['ytpos'],
                        data['Forcex'], data['Forcey']], axis=0).T
    cur_len = signals.shape[0]
    if cur_len < target_length:
        pad = np.zeros((target_length - cur_len, N_CHANNELS))
        signals = np.vstack([signals, pad])
    else:
        signals = signals[:target_length, :]
    return signals.astype(np.float32)

# ---- 3. Build fold datasets ----
def build_fold_data(fold_id):
    assigned_groups = set(group_to_fold.keys()) - dev_groups
    val_groups = {g for g, f in group_to_fold.items() if int(f) == fold_id}
    train_groups = assigned_groups - val_groups

    train_files = table.loc[table['parent_group'].isin(train_groups), 'file'].tolist()
    val_files   = table.loc[table['parent_group'].isin(val_groups), 'file'].tolist()

    print(f"Fold {fold_id}: train files {len(train_files)}, val files {len(val_files)}")

    def load_split(file_list, with_weights=False):
        X, w = [], []
        for fname in file_list:
            row = table[table['file'] == fname]
            if row.empty:
                fname_stem = os.path.splitext(fname)[0]
                row = table[table['file'].str.startswith(fname_stem)]
            if row.empty:
                continue
            row = row.iloc[0]
            npz_path = os.path.join(SIGNAL_DIR,
                                    f"dataset_{int(row['dataset'])}",
                                    fname.replace('.h5', '.npz'))
            X.append(load_signal(npz_path))
            w.append(float(row['sample_weight']))
        return np.array(X, dtype=np.float32), np.array(w, dtype=np.float32) if with_weights else np.array(X)

    X_tr, w_tr = load_split(train_files, with_weights=True)
    X_val, w_val = load_split(val_files, with_weights=True)
    return X_tr, w_tr, X_val, w_val

# Build the autoencoder
inputs = layers.Input(shape=(T, N_CHANNELS), name='signal_input')
x = SigKAN(100, 2, dropout=0.1)(inputs)           # (batch, T', 100)
x = layers.GlobalAveragePooling1D()(x)             # (batch, 100)
bottleneck = layers.Dense(LATENT_DIM, activation='relu', name='bottleneck')(x)

# Decoder
x = layers.Dense(256, activation='relu')(bottleneck)
x = layers.Dense(T * N_CHANNELS, activation='linear')(x)      # 2048 * 4 = 8192
reconstructed = layers.Reshape((T, N_CHANNELS))(x)

autoencoder = models.Model(inputs, reconstructed)
autoencoder.compile(optimizer='adam', loss='mse')

# Load data
X_tr, w_tr, X_val, w_val = build_fold_data(FOLD_ID)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
]

history = autoencoder.fit(
    X_tr, X_tr,                    # input = target
    sample_weight=w_tr,            # emphasise clean signals
    validation_data=(X_val, X_val, w_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save
os.makedirs(MODEL_DIR, exist_ok=True)
autoencoder.save(os.path.join(MODEL_DIR, f"autoencoder_fold{FOLD_ID}.keras"))
print(f"Autoencoder saved to {MODEL_DIR}")