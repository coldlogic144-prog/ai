import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image

# 1. Load saved model weights and biases (keeping them intact)
model_folder = Path("models")
model_folder.mkdir(exist_ok=True)
model_path = model_folder / "digit_model.npz"

if model_path.exists():
    saved_model = np.load(model_path)
    weight = saved_model["weights"]
    bias = saved_model["biases"]
    print("Saved weights loaded!")
else:
    print("No saved model found, initializing new weights...")
    weight = np.random.randn(10, 784) * 0.01
    bias = np.zeros(10)

# 2. Download MNIST dataset only if it doesn't already exist (fixes downloading every time)
url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)
file_path = data_folder / "mnist.npz"

if not file_path.exists():
    print("Downloading MNIST dataset...")
    urllib.request.urlretrieve(url, file_path)
    print("Download complete!")
    print("Saved to:", file_path)
else:
    print(f"Dataset already exists at {file_path}. Skipping download.")

# 3. Load dataset
data = np.load(file_path)
print("Dataset files:", data.files)

images = data["x_train"]
labels = data["y_train"]

image = images[0]

flattened_image = image.flatten()
normalized_image = flattened_image / 255.0

# 4. Forward pass prediction with existing saved weights
output = np.dot(weight, normalized_image) + bias
index = np.argmax(output)

print("Number is (predicted):", index)
print("What it was (actual):", labels[0])


# 5. Weight and Bias Updater Function
def update_weights_and_biases(weights, biases, x, y, learning_rate=0.1):
    """
    Updates weights and biases using Softmax & Cross-Entropy gradient descent.
    
    Creates and returns new copies so the original weights saved by you
    are NOT modified in place.
    
    Parameters:
        weights: np.ndarray of shape (10, 784)
        biases: np.ndarray of shape (10,)
        x: np.ndarray, single image (784,) or batch of images (N, 784)
        y: int or np.ndarray, true label (0-9) or batch of labels (N,)
        learning_rate: float, step size / eta (default: 0.1)
        
    Returns:
        new_weights, new_biases: updated weight and bias arrays
    """
    # Create copies so original weights/biases are untouched
    new_weights = weights.copy()
    new_biases = biases.copy()

    # Handle single sample (1D) vs batch (2D)
    if x.ndim == 1:
        x_batch = x.reshape(1, -1)
        y_batch = np.array([y])
    else:
        x_batch = x
        y_batch = np.array(y)

    num_samples = x_batch.shape[0]

    # Forward pass logits: z = x @ W.T + b  (shape: N x 10)
    z = np.dot(x_batch, new_weights.T) + new_biases

    # Softmax probabilities (numerically stable)
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    # One-hot target labels
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(num_samples), y_batch] = 1.0

    # Cross-entropy gradient w.r.t logits: dz = (probs - one_hot) / N
    dz = (probs - one_hot) / num_samples

    # Gradients w.r.t weights (dW) and biases (db)
    dw = np.dot(dz.T, x_batch)  # shape: (10, 784)
    db = np.sum(dz, axis=0)      # shape: (10,)

    # Gradient descent update
    new_weights -= learning_rate * dw
    new_biases -= learning_rate * db

    return new_weights, new_biases


# 6. Run updater function
n = 0.1
new_weight, new_bias = update_weights_and_biases(
    weights=weight,
    biases=bias,
    x=normalized_image,
    y=labels[0],
    learning_rate=n
)

# 7. Save updated weights to a separate file so your original weights remain safe
updated_model_path = model_folder / "digit_model_updated.npz"
np.savez(
    updated_model_path,
    weights=new_weight,
    biases=new_bias
)

print(f"Updated weights and biases saved to: {updated_model_path}")
print("Your original 'digit_model.npz' was left completely untouched!")