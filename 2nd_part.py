"""
NYC Taxi Trip Duration Prediction - Compact & Efficient Implementation
-----------------------------------------------------------------------
Trains 3+ model configurations with early stopping and evaluates on test set.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from neural_net_library_xor import Sequential, Linear, ReLU

np.random.seed(42)

# ============= DATA & FEATURES =============

def engineer_features(df):
    """Add temporal and spatial features - optimized."""
    df = df.copy()
    dt = pd.to_datetime(df["pickup_datetime"])
    
    # Temporal - vectorized
    df["pickup_hour"] = dt.dt.hour
    df["pickup_dayofweek"] = dt.dt.dayofweek
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(np.int8)
    
    # Extract coordinates once
    lat1, lon1 = df["pickup_latitude"].values, df["pickup_longitude"].values
    lat2, lon2 = df["dropoff_latitude"].values, df["dropoff_longitude"].values
    
    # Haversine distance - fully vectorized
    lat1_r, lon1_r, lat2_r, lon2_r = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2_r - lat1_r, lon2_r - lon1_r
    a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
    df["dist_km"] = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    
    # Bearing - vectorized
    y = np.sin(dlon) * np.cos(lat2_r)
    x = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    df["bearing"] = np.degrees(np.arctan2(y, x))
    
    # Binary flag
    df["store_fwd"] = (df["store_and_fwd_flag"] == "Y").astype(np.int8)
    
    return df.drop(columns=["id", "pickup_datetime", "dropoff_datetime", "store_and_fwd_flag"], errors="ignore")

def load_and_preprocess(path="nyc_taxi_data.npy", val_split=0.1):
    """Load, engineer features, split, and normalize - optimized."""
    dataset = np.load(path, allow_pickle=True).item()
    
    print("Engineering features...")
    X_tr = engineer_features(dataset["X_train"]).values.astype(np.float32)
    X_te = engineer_features(dataset["X_test"]).values.astype(np.float32)
    
    # Log transform targets
    y_tr_log = np.log1p(dataset["y_train"].values).astype(np.float32)
    y_te_log = np.log1p(dataset["y_test"].values).astype(np.float32)
    y_te_orig = dataset["y_test"].values.astype(np.float32)
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr_log, test_size=val_split, random_state=42
    )
    
    # Normalize using training stats (compute once)
    mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = (X_train.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
    
    # In-place normalization for memory efficiency
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_te - mean) / std
    
    return X_train, y_train, X_val, y_val, X_test, y_te_log, y_te_orig

# ============= MODEL & TRAINING =============

def build_model(input_dim, hidden_layers, seed=42):
    """Build regression network: Input → [Linear→ReLU]×n → Linear."""
    rng = np.random.RandomState(seed)
    layers = []
    prev = input_dim
    for h in hidden_layers:
        layers.extend([Linear(prev, h, rng=rng), ReLU()])
        prev = h
    layers.append(Linear(prev, 1, rng=rng))
    return Sequential(*layers)

def train(model, X_tr, y_tr, X_val, y_val, lr=1e-3, bs=1024, epochs=60, patience=3, verbose_every=5):
    """Train with MSE loss and early stopping. Print every N epochs for speed."""
    n = len(X_tr)
    best_val, best_ep, wait = float('inf'), -1, 0
    tr_loss, val_loss = [], []
    
    # Pre-reshape validation target once
    y_val_reshaped = y_val.reshape(-1, 1)
    
    for ep in range(epochs):
        # Shuffle once per epoch
        idx = np.random.permutation(n)
        X_tr_shuffled, y_tr_shuffled = X_tr[idx], y_tr[idx]
        ep_loss = 0
        
        for i in range(0, n, bs):
            xb, yb = X_tr_shuffled[i:i+bs], y_tr_shuffled[i:i+bs].reshape(-1, 1)
            pred = model.forward(xb)
            
            # Efficient loss and gradient
            diff = pred - yb
            loss = (diff * diff).mean()
            grad = 2 * diff / len(xb)
            
            model.backward(grad)
            
            # Vectorized parameter update
            for p, g in zip(model.params.values(), model.grads.values()):
                p -= lr * g
            
            ep_loss += loss * len(xb)
        
        ep_loss /= n
        
        # Validation without creating new arrays
        val_pred = model.forward(X_val)
        vloss = ((val_pred - y_val_reshaped) ** 2).mean()
        
        tr_loss.append(float(ep_loss))
        val_loss.append(float(vloss))
        
        # Print only every N epochs
        if (ep + 1) % verbose_every == 0 or ep == 0:
            print(f"Epoch {ep+1:3d} | Train: {ep_loss:.6f} | Val: {vloss:.6f}")
        
        # Early stopping check
        if vloss < best_val - 1e-4:
            best_val, best_ep, wait = vloss, ep, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {ep+1}, best: {best_val:.6f} (epoch {best_ep+1})")
                break
    
    return tr_loss, val_loss, float(best_val), best_ep

# ============= EVALUATION & PLOTTING =============

def evaluate(model, X_test, y_log, y_orig):
    """Compute test metrics - optimized."""
    pred_log = model.forward(X_test).ravel()
    pred_sec = np.expm1(pred_log)
    diff = pred_sec - y_orig
    mae = np.abs(diff).mean()
    rmse = np.sqrt((diff * diff).mean())
    return float(mae), float(rmse)

def plot_curves(tr, val, name, best_ep):
    """Plot training/validation curves - non-blocking."""
    plt.figure(figsize=(8, 5))
    plt.plot(tr, label='Train', linewidth=2)
    plt.plot(val, label='Val', linewidth=2)
    plt.axvline(best_ep+1, color='g', linestyle='--', label=f'Best ({best_ep+1})')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title(f'{name}')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f'{name}_loss.png', dpi=200, bbox_inches='tight')  # Lower DPI for speed
    plt.close()

def plot_comparison(results):
    """Bar charts comparing all models - non-blocking."""
    names = list(results.keys())
    vals = [results[n]['val_loss'] for n in names]
    maes = [results[n]['test_mae'] for n in names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.bar(names, vals, color='steelblue', alpha=0.8)
    ax1.set_ylabel('Best Val Loss')
    ax1.set_title('Validation Performance')
    ax1.grid(alpha=0.3, axis='y')
    for i, v in enumerate(vals):
        ax1.text(i, v, f'{v:.5f}', ha='center', va='bottom', fontsize=9)
    
    ax2.bar(names, maes, color='coral', alpha=0.8)
    ax2.set_ylabel('Test MAE (sec)')
    ax2.set_title('Test Performance')
    ax2.grid(alpha=0.3, axis='y')
    for i, v in enumerate(maes):
        ax2.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=200, bbox_inches='tight')  # Lower DPI
    plt.close()

# ============= MAIN =============

if __name__ == "__main__":
    print("="*60)
    print("NYC TAXI DURATION PREDICTION")
    print("="*60)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test_log, y_test = load_and_preprocess()
    input_dim = X_train.shape[1]
    print(f"Data: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test | Features: {input_dim}\n")
    
    # Configurations
    configs = [
        {"name": "Small_64x32", "layers": [64, 32], "lr": 1e-3},
        {"name": "Medium_128x64x32", "layers": [128, 64, 32], "lr": 1e-3},
        {"name": "Large_256x128x64", "layers": [256, 128, 64], "lr": 5e-4},
    ]
    
    results = {}
    
    # Train all configs
    for i, cfg in enumerate(configs, 1):
        print(f"{'='*60}")
        print(f"CONFIG {i}/{len(configs)}: {cfg['name']} {cfg['layers']}")
        print(f"{'='*60}")
        
        model = build_model(input_dim, cfg['layers'], seed=42+i)
        tr_loss, val_loss, best_val, best_ep = train(
            model, X_train, y_train, X_val, y_val,
            lr=cfg['lr'], bs=1024, epochs=60, patience=3, verbose_every=5
        )
        
        mae, rmse = evaluate(model, X_test, y_test_log, y_test)
        print(f"Test MAE: {mae:.2f}s | RMSE: {rmse:.2f}s")
        
        results[cfg['name']] = {'val_loss': best_val, 'test_mae': mae, 'test_rmse': rmse}
        plot_curves(tr_loss, val_loss, cfg['name'], best_ep)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Val Loss':>12} {'Test MAE':>12}")
    print("-"*60)
    for name, res in results.items():
        print(f"{name:<20} {res['val_loss']:>12.6f} {res['test_mae']:>12.2f}s")
    
    best = min(results.items(), key=lambda x: x[1]['val_loss'])
    print(f"\n★ Best: {best[0]} (Val: {best[1]['val_loss']:.6f}, MAE: {best[1]['test_mae']:.2f}s)")
    
    plot_comparison(results)
    print("\n✓ Complete! Plots saved (loss curves + comparison.png)")