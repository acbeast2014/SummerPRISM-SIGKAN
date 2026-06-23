# train_sigkan.py
import os, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sigkan import SigKAN          # official package


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
table['parent_group'] = table['parent_group'].astype(str)
with open(os.path.join(METADATA_DIR, "phase3_folds.json"), "r") as f:
    folds_json = json.load(f)

group_to_fold = folds_json['group_to_fold']
dev_groups = set(map(str, folds_json.get('dev_groups', [])))

# ---- 2. Signal loader ----
def load_signal(file_path, target_length=T):
    data = np.load(file_path)
    # shape (4, L) → (L, 4)
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
    Uses group_to_fold from phase3_folds.json.
    - Validation = all files whose parent_group is assigned to fold_id.
    - Train = all other groups (except dev_groups).
    """
    assigned_groups = set(group_to_fold.keys()) - dev_groups
    val_groups = {g for g, f in group_to_fold.items() if int(f) == fold_id}
    train_groups = assigned_groups - val_groups

    train_files = table.loc[table['parent_group'].isin(train_groups), 'file'].tolist()
    val_files   = table.loc[table['parent_group'].isin(val_groups), 'file'].tolist()

    print(f"Fold {fold_id}: train groups={len(train_groups)}, val groups={len(val_groups)}")
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    def load_split(file_list):
        X, y, w = [], [], []
        for fname in file_list:
            row = table[table['file'] == fname]
            if row.empty:
                fname_stem = os.path.splitext(fname)[0]
                row = table[table['file'].str.startswith(fname_stem)]
            if row.empty:
                print(f"Warning: {fname} not found, skipping")
                continue
            row = row.iloc[0]
            npz_path = os.path.join(SIGNAL_DIR,
                                    f"dataset_{int(row['dataset'])}",
                                    fname.replace('.h5', '.npz'))
            X.append(load_signal(npz_path))
            y.append(int(row['label']))
            w.append(float(row['sample_weight']))
        return (np.array(X, dtype=np.float32),
                np.array(y, dtype=np.int32),
                np.array(w, dtype=np.float32))

    X_tr, y_tr, w_tr = load_split(train_files)
    X_val, y_val, w_val = load_split(val_files)
    return (X_tr, y_tr, w_tr), (X_val, y_val, w_val)

# Load data
(X_tr, y_tr, w_tr), (X_val, y_val, w_val) = build_fold_dataset(FOLD_ID)

# Build the model 
model = models.Sequential([
    layers.Input(shape=(T, N_CHANNELS)),
    SigKAN(100, 2, dropout=0.1),        # returns (batch, T?, 100)
    layers.Flatten(),                   # → (batch, ...)
    layers.Dense(100, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
    jit_compile=False                   # safe for now
)

callbacks = [
    EarlyStopping(patience=20, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)
]

history = model.fit(
    X_tr, y_tr,
    sample_weight=w_tr,
    validation_data=(X_val, y_val, w_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, f"sigkan_fold{FOLD_ID}.keras"))

test_loss, test_acc = model.evaluate(X_val, y_val, sample_weight=w_val, verbose=0)
print(f"Validation accuracy: {test_acc:.4f}")