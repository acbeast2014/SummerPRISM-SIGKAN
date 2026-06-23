import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---- Import official SigKAN layer ----
# Adjust the import path to wherever you placed the sigkan module
from custom_layers.sigkan import SigKAN

# ======================== CONFIG ========================
T = 2048                 # fixed signal length (power of 2 recommended)
N_CHANNELS = 4           # xtpos, ytpos, Forcex, Forcey
BATCH_SIZE = 64
EPOCHS = 150
FOLD_ID = 0              # which fold to use as test set (or loop over all)
SIGNAL_DIR = "data/phase2_signals"
METADATA_DIR = "results/phase3"
MODEL_DIR = "models/sigkan"
# =========================================================

# ---- 1. Load metadata and folds ----
table = pd.read_csv(os.path.join(METADATA_DIR, "phase3_dataset.csv"))
with open(os.path.join(METADATA_DIR, "phase3_folds.json"), "r") as f:
    folds = json.load(f)

# ---- 2. Signal loader ----
def load_signal(file_path, target_length=T):
    data = np.load(file_path)
    # Stack channels: (4, original_len) -> (original_len, 4)
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
def build_fold_dataset(fold_id, table, folds, signal_dir, target_length=T):
    """
    Returns (X_train, y_train, w_train), (X_val, y_val, w_val)
    """
    fold_info = folds['folds'][str(fold_id)]
    train_files = fold_info['train']
    val_files   = fold_info['val']

    def load_split(file_list):
        X, y, w = [], [], []
        for fname in file_list:
            # Find the row in the table (file name may include .h5)
            row = table[table['file'] == fname]
            if row.empty:
                # try without extension
                fname_stem = os.path.splitext(fname)[0]
                row = table[table['file'].str.startswith(fname_stem)]
            if row.empty:
                raise FileNotFoundError(f"Could not find file {fname} in phase3_dataset.csv")
            row = row.iloc[0]
            npz_path = os.path.join(
                signal_dir,
                f"dataset_{int(row['dataset'])}",
                fname.replace('.h5', '.npz')
            )
            X.append(load_signal(npz_path, target_length))
            y.append(int(row['label']))
            w.append(float(row['sample_weight']))
        return np.array(X), np.array(y, dtype=np.int32), np.array(w, dtype=np.float32)

    X_train, y_train, w_train = load_split(train_files)
    X_val, y_val, w_val = load_split(val_files)
    return (X_train, y_train, w_train), (X_val, y_val, w_val)

# ---- 4. Build the SigKAN classifier ----
def build_sigkan_classifier(input_shape=(T, N_CHANNELS),
                           sigkan_units=100,
                           sigkan_depth=2,
                           dropout_rate=0.1):
    inputs = layers.Input(shape=input_shape, name='signal_input')
    x = SigKAN(sigkan_units, sigkan_depth, dropout=dropout_rate, name='sig_kan')(inputs)
    # The official layer likely returns 2D (batch, sigkan_units * sequence_length) or 3D?
    # Based on the original snippet, it flattens after SigKAN, so we'll do that.
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ---- 5. Main training loop ----
# Load data for the chosen fold
(X_tr, y_tr, w_tr), (X_val, y_val, w_val) = build_fold_dataset(
    FOLD_ID, table, folds, SIGNAL_DIR
)

print(f"Train samples: {X_tr.shape[0]}, Val samples: {X_val.shape[0]}")
print(f"Label distribution (train): {np.bincount(y_tr)}")
print(f"Label distribution (val):   {np.bincount(y_val)}")

# Build model
model = build_sigkan_classifier()
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
]

# Train with sample weights
history = model.fit(
    X_tr, y_tr,
    sample_weight=w_tr,
    validation_data=(X_val, y_val, w_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ---- 6. Save model ----
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, f"sigkan_fold{FOLD_ID}.h5"))

# ---- 7. Quick evaluation ----
test_loss, test_acc = model.evaluate(X_val, y_val, sample_weight=w_val, verbose=0)
print(f"Fold {FOLD_ID} validation accuracy: {test_acc:.4f}")