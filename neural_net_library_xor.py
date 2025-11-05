"""
Neural Network Library - Assignment Implementation
---------------------------------------------------
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

Array = np.ndarray

# weight initialization
def glorot_uniform(fan_in: int, fan_out: int, rng: np.random.RandomState) -> Array:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_out, fan_in))

# base layer (blue print for all layers[linear, Sigmoidal ReLu, TanH, BinaryCrossEntropy, Sequential])
class Layer:
    def forward(self, x: Array) -> Array:
        raise NotImplementedError("Subclasses must implement forward()")
    
    def backward(self, grad_output: Array) -> Array:
        raise NotImplementedError("Subclasses must implement backward()")
    
    @property
    def params(self) -> Dict[str, Array]:
        return {}
    
    @property
    def grads(self) -> Dict[str, Array]:
        return {}


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, 
                 rng: Optional[np.random.RandomState] = None):
        """Initialize linear layer with Xavier initialization."""
        self.in_features = in_features
        self.out_features = out_features
        self.rng = rng or np.random.RandomState(42)
        
        # Initialize weights using Xavier/Glorot initialization
        self.w = glorot_uniform(in_features, out_features, self.rng)  # (h, d)
        self.b = np.zeros(out_features)  # (h,)
        
        # Pre-allocate gradient arrays for efficiency
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        
        # Cache for backward pass
        self._x_cache: Optional[Array] = None
    
    def forward(self, x: Array) -> Array:
        # Cache input for use in backward pass
        self._x_cache = x
        
        # Compute linear transformation
        return x @ self.w.T + self.b
    
    def backward(self, grad_output: Array) -> Array:
        assert self._x_cache is not None, "Must call forward() before backward()"
        
        # Gradient with respect to weights: dL/dw = grad_output^T @ x
        np.dot(grad_output.T, self._x_cache, out=self.dw)
        
        # Gradient with respect to bias: dL/db = sum over batch dimension
        np.sum(grad_output, axis=0, out=self.db)
        
        # Gradient with respect to input (for previous layer)
        grad_input = grad_output @ self.w
        
        return grad_input
    
    @property
    def params(self) -> Dict[str, Array]:
        """Return learnable parameters."""
        return {"w": self.w, "b": self.b}
    
    @property
    def grads(self) -> Dict[str, Array]:
        """Return gradients of learnable parameters."""
        return {"w": self.dw, "b": self.db}

# Activation layer
class Sigmoid(Layer):
    def __init__(self):
        """Initialize sigmoid activation."""
        self._output_cache: Optional[Array] = None
    
    def forward(self, x: Array) -> Array:
        # Clip input to prevent overflow in exp
        x_clipped = np.clip(x, -500, 500)
        
        # Compute sigmoid and cache for backward pass
        self._output_cache = 1.0 / (1.0 + np.exp(-x_clipped))
        
        return self._output_cache
    
    def backward(self, grad_output: Array) -> Array:
        assert self._output_cache is not None, "Must call forward() before backward()"
        
        # Reuse cached sigmoid output
        sigma = self._output_cache
        
        # Compute gradient: grad_output * σ(x) * (1 - σ(x))
        return grad_output * sigma * (1.0 - sigma)


class ReLU(Layer):
    def __init__(self):
        """Initialize ReLU activation."""
        self._mask_cache: Optional[Array] = None
    
    def forward(self, x: Array) -> Array:
        # Cache mask of positive values for backward pass
        self._mask_cache = (x > 0)
        
        # Apply ReLU: max(0, x)
        return np.maximum(x, 0.0)
    
    def backward(self, grad_output: Array) -> Array:
        assert self._mask_cache is not None, "Must call forward() before backward()"
        
        # Gradient passes through where input was positive
        return grad_output * self._mask_cache


class Tanh(Layer):
    def __init__(self):
        """Initialize Tanh activation."""
        self._output_cache: Optional[Array] = None
    
    def forward(self, x: Array) -> Array:
        # Compute tanh and cache for backward pass
        self._output_cache = np.tanh(x)
        return self._output_cache
    
    def backward(self, grad_output: Array) -> Array:
        assert self._output_cache is not None, "Must call forward() before backward()"
        
        # Reuse cached tanh output
        tanh_x = self._output_cache
        
        # Compute gradient: grad_output * (1 - tanh^2(x))
        return grad_output * (1.0 - tanh_x * tanh_x)

# Loss Function
class BinaryCrossEntropy(Layer):
    def __init__(self, eps: float = 1e-12):
        """Initialize binary cross-entropy loss."""
        self.eps = eps
        self._p_cache: Optional[Array] = None
        self._y_cache: Optional[Array] = None
        self._n_samples: int = 0
    
    def forward(self, p: Array, y: Array) -> float:
        # Clip predictions for numerical stability
        p_clipped = np.clip(p, self.eps, 1.0 - self.eps)
        
        # Cache for backward pass
        self._p_cache = p_clipped
        self._y_cache = y
        self._n_samples = y.shape[0]
        
        # Compute binary cross-entropy
        loss = -np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped))
        
        return float(loss)
    
    def backward(self, grad_output: float = 1.0) -> Array:
        assert self._p_cache is not None, "Must call forward() before backward()"
        
        p = self._p_cache
        y = self._y_cache
        n = self._n_samples
        
        # Compute gradient
        grad = (p - y) / (p * (1.0 - p)) / n
        
        return grad_output * grad


class Sequential(Layer):
    def __init__(self, *layers: Layer):
        """Initialize sequential container with given layers."""
        self.layers: List[Layer] = list(layers)
    
    def add(self, layer: Layer) -> None:
        self.layers.append(layer)
    
    def forward(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output: Array) -> Array:
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output
    
    @property
    def params(self) -> Dict[str, Array]:
        all_params = {}
        for idx, layer in enumerate(self.layers):
            layer_params = layer.params
            for name, value in layer_params.items():
                all_params[f"{idx}.{name}"] = value
        return all_params
    
    @property
    def grads(self) -> Dict[str, Array]:
        all_grads = {}
        for idx, layer in enumerate(self.layers):
            layer_grads = layer.grads
            for name, value in layer_grads.items():
                all_grads[f"{idx}.{name}"] = value
        return all_grads
    
    def save_weights(self, filepath: str) -> None:
        # Ensure .npz extension
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        
        # Save all parameters
        np.savez(filepath, **self.params)
        print(f"Weights saved to '{filepath}'")
    
    def load_weights(self, filepath: str) -> None:
        # Ensure .npz extension
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        
        # Load parameters
        data = np.load(filepath)
        
        # Update each layer's parameters
        for idx, layer in enumerate(self.layers):
            layer_params = layer.params
            for name in layer_params.keys():
                key = f"{idx}.{name}"
                if key in data:
                    layer_params[name][:] = data[key]
                else:
                    raise KeyError(f"Parameter '{key}' not found in file")
        
        print(f"Weights loaded from '{filepath}'")

# Optimizer
class SGD:
    def __init__(self, params: Dict[str, Array], lr: float = 0.1):
        """Initialize SGD optimizer."""
        self.params = params
        self.lr = lr
    
    def step(self, grads: Dict[str, Array]) -> None:
        for key in self.params:
            self.params[key] -= self.lr * grads[key]

# Training funtion
def train(model: Sequential, loss_fn: BinaryCrossEntropy, 
          X: Array, y: Array, epochs: int = 5000, lr: float = 0.1,
          verbose_every: int = 500) -> tuple:
    optimizer = SGD(model.params, lr=lr)
    losses = []
    accuracies = []
    
    for epoch in range(1, epochs + 1):
        # Forward pass
        predictions = model.forward(X)
        loss = loss_fn.forward(predictions, y)
        
        # Backward pass
        grad = loss_fn.backward()
        model.backward(grad)
        
        # Update parameters
        optimizer.step(model.grads)
        
        # Compute accuracy
        acc = np.mean((predictions >= 0.5) == y)
        
        # Track metrics
        losses.append(loss)
        accuracies.append(acc)
        
        # Print progress
        if verbose_every and epoch % verbose_every == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss:.6f} | Accuracy: {acc:.3f}")
    
    return losses, accuracies


# ============================================================================
#                           XOR PROBLEM TESTING
# ============================================================================

def make_xor_dataset() -> tuple:
    X = np.array([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])
    y = np.array([[0.],
                  [1.],
                  [1.],
                  [0.]])
    return X, y


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    np.set_printoptions(precision=4, suppress=True)
    
    # Prepare XOR dataset
    X, y = make_xor_dataset()
    loss_fn = BinaryCrossEntropy()
    
    print("\n" + "="*70)
    print("NEURAL NETWORK LIBRARY - XOR PROBLEM")
    print("="*70)
    print("\nArchitecture: 2 input nodes → 2 hidden nodes → 1 output node")
    print("Dataset: XOR (4 samples, 2 features)")
    
    # ========================================================================
    # Train with SIGMOID activation
    # ========================================================================
    print("\n" + "-"*70)
    print("EXPERIMENT 1: Training with SIGMOID hidden layer activation")
    print("-"*70)
    
    rng_sigmoid = np.random.RandomState(123)
    model_sigmoid = Sequential(
        Linear(2, 2, rng=rng_sigmoid),
        Sigmoid(),
        Linear(2, 1, rng=rng_sigmoid),
        Sigmoid()
    )
    
    losses_sig, accs_sig = train(
        model_sigmoid, loss_fn, X, y,
        epochs=8000, lr=0.5, verbose_every=1000
    )
    
    final_acc_sig = accs_sig[-1]
    final_loss_sig = losses_sig[-1]
    print(f"\nFinal Results (Sigmoid):")
    print(f"  Accuracy: {final_acc_sig:.3f}")
    print(f"  Loss: {final_loss_sig:.6f}")
    
    # ========================================================================
    # Train with TANH activation
    # ========================================================================
    print("\n" + "-"*70)
    print("EXPERIMENT 2: Training with TANH hidden layer activation")
    print("-"*70)
    
    rng_tanh = np.random.RandomState(456)
    model_tanh = Sequential(
        Linear(2, 2, rng=rng_tanh),
        Tanh(),
        Linear(2, 1, rng=rng_tanh),
        Sigmoid()  # Output layer uses sigmoid for binary classification
    )
    
    losses_tanh, accs_tanh = train(
        model_tanh, loss_fn, X, y,
        epochs=8000, lr=0.5, verbose_every=1000
    )
    
    final_acc_tanh = accs_tanh[-1]
    final_loss_tanh = losses_tanh[-1]
    print(f"\nFinal Results (Tanh):")
    print(f"  Accuracy: {final_acc_tanh:.3f}")
    print(f"  Loss: {final_loss_tanh:.6f}")
    
    # ========================================================================
    # Select and save best model
    # ========================================================================
    print("\n" + "="*70)
    print("SAVING BEST MODEL")
    print("="*70)
    
    # Select model with better accuracy (or lower loss if tied)
    if final_acc_tanh > final_acc_sig:
        best_model = model_tanh
        best_name = "Tanh"
    elif final_acc_sig > final_acc_tanh:
        best_model = model_sigmoid
        best_name = "Sigmoid"
    else:
        best_model = model_tanh if final_loss_tanh < final_loss_sig else model_sigmoid
        best_name = "Tanh" if final_loss_tanh < final_loss_sig else "Sigmoid"
    
    print(f"\nBest performing model: {best_name} activation")
    
    # Save the best model
    best_model.save_weights("XOR_solved.w")
    
    # ========================================================================
    # Verify solution
    # ========================================================================
    print("\n" + "="*70)
    print("VERIFICATION - Model Predictions")
    print("="*70)
    
    predictions = best_model.forward(X)
    predicted_labels = (predictions >= 0.5).astype(int)
    
    print("\nInput | Predicted Prob | Predicted Label | True Label")
    print("-" * 60)
    for i in range(len(X)):
        x_str = f"[{X[i, 0]:.0f}, {X[i, 1]:.0f}]"
        prob = predictions[i, 0]
        pred = predicted_labels[i, 0]
        true = int(y[i, 0])
        status = "✓" if pred == true else "✗"
        print(f"{x_str:7s} | {prob:14.4f} | {pred:15d} | {true:10d}  {status}")
    
    all_correct = np.all(predicted_labels == y.astype(int))
    print(f"\n{'='*70}")
    print(f"XOR Problem Solved: {all_correct}")
    print(f"Best Model: {best_name} activation in hidden layer")
    print(f"{'='*70}\n")

x = np.linspace(-5, 5, 500)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)

plt.figure(figsize=(8,5))
plt.plot(x, sigmoid, label='Sigmoid', linewidth=2)
plt.plot(x, tanh, label='Tanh', linewidth=2)
plt.title("Sigmoid vs Tanh Activation Functions")
plt.xlabel("Input")
plt.ylabel("Activation Output")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig("sigmoid_vs_tanh_activation.png", dpi=200, bbox_inches='tight')
plt.show()