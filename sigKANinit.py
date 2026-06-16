import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from custom_layers.sigkan import SigKAN          # adjust path as needed
import matplotlib.pyplot as plt                  # for weight inspection later

# ---------------------------
# 1. CONFIGURATION
# ---------------------------
T = 2048               # fixed signal length
n_channels = 4
latent_dim = 100       # SigKAN output dimension
bottleneck = 32        # compressed representation size
signal_dir = 'data/phase2_signals'

# ---------------------------
# 2. DATA LOADING (only signals needed for autoencoder)
# ---------------------------
def load_signal(file_path, target_length=T):
    data = np.load(file_path)
    signals = np.stack([data['xtpos'], data['ytpos'],
                        data['Forcex'], data['Forcey']], axis=0)  # (4, L)
    signals = signals.T  # (L, 4)
    cur_len = signals.shape[0]
    if cur_len < target_length:
        pad = np.zeros((target_length - cur_len, n_channels))
        signals = np.vstack([signals, pad])
    else:
        signals = signals[:target_length, :]
    return signals.astype(np.float32)

def build_autoencoder_data(fold_id, table, folds, signal_dir, target_length=T):
    """Return (X_train, w_train, X_val, w_val) for unsupervised reconstruction."""
    fold = folds['folds'][str(fold_id)]
    train_files = fold['train']
    val_files   = fold['val']

    def load_split(file_list):
        X, w = [], []
        for f in file_list:
            row = table[table['file'] == f].iloc[0]
            npz_path = f"{signal_dir}/dataset_{row['dataset']}/{f.replace('.h5','.npz')}"
            X.append(load_signal(npz_path, target_length))
            w.append(row['sample_weight'])
        return np.array(X), np.array(w)
    
    X_train, w_train = load_split(train_files)
    X_val,   w_val   = load_split(val_files)
    return X_train, w_train, X_val, w_val

# Load metadata
table = pd.read_csv('results/phase3/phase3_dataset.csv')
with open('results/phase3/phase3_folds.json', 'r') as f:
    folds = json.load(f)

fold_id = 0
X_tr, w_tr, X_val, w_val = build_autoencoder_data(
    fold_id, table, folds, signal_dir
)

# ---------------------------
# 3. AUTOENCODER MODEL
# ---------------------------
inputs = layers.Input(shape=(T, n_channels), name='signal_input')
x = SigKAN(latent_dim, 2, dropout=0.1, name='sig_kan')(inputs)
# Assuming SigKAN returns (batch, T', latent_dim) -> pool time
x = layers.GlobalAveragePooling1D()(x)
bottleneck_tensor = layers.Dense(bottleneck, activation='relu', name='bottleneck')(x)

# Decoder
x = layers.Dense(256, activation='relu')(bottleneck_tensor)
x = layers.Dense(T * n_channels, activation='linear')(x)
reconstructed = layers.Reshape((T, n_channels), name='reconstructed')(x)

autoencoder = models.Model(inputs, reconstructed)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# ---------------------------
# 4. TRAINING WITH SAMPLE WEIGHTS
# ---------------------------
history = autoencoder.fit(
    X_tr, X_tr,
    sample_weight=w_tr,                  # emphasise clean signals
    validation_data=(X_val, X_val, w_val),
    batch_size=32,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7)
    ],
    verbose=1
)

# ---------------------------
# 5. INSPECT KAN WEIGHTS (Example)
# ---------------------------
# Get the SigKAN layer by name (we gave it name='sig_kan')
sigkan_layer = autoencoder.get_layer('sig_kan')
# The weights structure depends on the exact SigKAN implementation.
# Typically you'll find spline control points in a trainable variable.
print("SigKAN weights:")
for w in sigkan_layer.weights:
    print(w.name, w.shape)

# If the layer stores grid and coefficients, you can plot them.
# For demonstration, assume we have a weight 'coef' of shape (in_features, out_features, grid_size).
# You'll need to adapt this to the actual SigKAN API.
try:
    # This is hypothetical – adjust to the real attribute names.
    coef = sigkan_layer.coef.numpy()
    grid = sigkan_layer.grid.numpy()
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for i in range(3):
        for j in range(3):
            axes[i, j].plot(grid, coef[i, j])
            axes[i, j].set_title(f'in {i} → out {j}')
    plt.tight_layout()
    plt.savefig('kan_splines.png')
    print('Spline plot saved.')
except AttributeError:
    print('SigKAN layer does not expose coef/grid directly – check its documentation.')