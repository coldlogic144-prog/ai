"""
ColdLogic - Retrain Digit Model with User Labeled Dataset
Reads exported JSON dataset from the web app, runs Softmax Cross-Entropy gradient descent,
and updates both the NumPy model file (.npz) and the static web model files (model_data.js & model.json).
"""

import sys
import json
import glob
from pathlib import Path
import numpy as np


def load_dataset(json_path=None):
    """Finds and loads the user dataset exported from the web interface."""
    if json_path and Path(json_path).exists():
        target = Path(json_path)
    else:
        # Search for any exported mnist_user_dataset*.json in current directory or Downloads
        candidates = sorted(glob.glob("mnist_user_dataset*.json"), reverse=True)
        if not candidates:
            # Check user's Downloads folder as convenience
            home_downloads = Path.home() / "Downloads"
            candidates = sorted(glob.glob(str(home_downloads / "mnist_user_dataset*.json")), reverse=True)

        if not candidates:
            print("[!] No dataset file found. Please export 'mnist_user_dataset.json' from the website first.")
            print("    Place the file in this folder and rerun: python train_user_data.py")
            return None, None

        target = Path(candidates[0])

    print(f"[+] Loading labeled dataset from: {target.name}")
    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    if not samples:
        print("[!] Dataset contains 0 samples. Label some digits on the website first!")
        return None, None

    x_list = []
    y_list = []
    for s in samples:
        x_list.append(s["vector"])
        y_list.append(int(s["label"]))

    X = np.array(x_list, dtype=np.float32)  # shape: (N, 784)
    Y = np.array(y_list, dtype=np.int64)    # shape: (N,)

    print(f"[+] Successfully loaded {len(X)} user-labeled samples.")
    return X, Y


def update_weights_and_biases(weights, biases, x, y, learning_rate=0.08):
    """
    Softmax Cross-Entropy gradient descent step (matching new.py).
    """
    new_weights = weights.copy()
    new_biases = biases.copy()

    if x.ndim == 1:
        x_batch = x.reshape(1, -1)
        y_batch = np.array([y])
    else:
        x_batch = x
        y_batch = np.array(y)

    num_samples = x_batch.shape[0]

    # Forward logits: z = x @ W.T + b  (shape: N x 10)
    z = np.dot(x_batch, new_weights.T) + new_biases

    # Softmax probabilities
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    # One-hot target labels
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(num_samples), y_batch] = 1.0

    # Cross-entropy gradients
    dz = (probs - one_hot) / num_samples
    dw = np.dot(dz.T, x_batch)  # (10, 784)
    db = np.sum(dz, axis=0)     # (10,)

    # Gradient descent update
    new_weights -= learning_rate * dw
    new_biases -= learning_rate * db

    return new_weights, new_biases


def evaluate(weights, biases, X, Y):
    """Calculates accuracy and average cross-entropy loss."""
    z = np.dot(X, weights.T) + biases
    preds = np.argmax(z, axis=1)
    acc = np.mean(preds == Y) * 100.0
    return acc, preds


def export_web_assets(weights, biases):
    """Updates model_data.js and model.json so the website uses the new weights immediately."""
    model_dict = {
        "weights": weights.tolist(),
        "biases": biases.tolist()
    }

    # Update model.json
    with open("model.json", "w", encoding="utf-8") as f:
        json.dump(model_dict, f)
    print("[+] Updated model.json")

    # Update model_data.js (for direct file:// and GitHub Pages loading without CORS fetch)
    js_content = f"// ColdLogic - MNIST Digit Recognizer Fine-Tuned Weights & Biases\nwindow.MODEL_DATA = {json.dumps(model_dict)};\n"
    with open("model_data.js", "w", encoding="utf-8") as f:
        f.write(js_content)
    print("[+] Updated model_data.js")


def main():
    json_arg = sys.argv[1] if len(sys.argv) > 1 else None
    X, Y = load_dataset(json_arg)
    if X is None:
        return

    # Load baseline model
    model_path = Path("models") / "digit_model.npz"
    if model_path.exists():
        saved = np.load(model_path)
        weights = saved["weights"]
        biases = saved["biases"]
        print(f"[+] Loaded baseline weights from: {model_path}")
    else:
        print("[!] models/digit_model.npz not found, using random initialization.")
        weights = np.random.randn(10, 784) * 0.01
        biases = np.zeros(10)

    # Initial accuracy on user samples
    initial_acc, _ = evaluate(weights, biases, X, Y)
    print(f"[*] Initial Accuracy on user samples: {initial_acc:.1f}%")

    # Fine-tuning loop
    epochs = 15
    lr = 0.08
    print(f"[*] Training on {len(X)} user samples for {epochs} epochs (learning rate: {lr})...")

    cur_weights = weights.copy()
    cur_biases = biases.copy()

    for epoch in range(1, epochs + 1):
        cur_weights, cur_biases = update_weights_and_biases(cur_weights, cur_biases, X, Y, learning_rate=lr)

    final_acc, final_preds = evaluate(cur_weights, cur_biases, X, Y)
    print(f"[+] Final Accuracy on user samples: {final_acc:.1f}% (+{final_acc - initial_acc:+.1f}%)")

    # Save to models/digit_model_updated.npz
    out_path = Path("models") / "digit_model_updated.npz"
    np.savez(out_path, weights=cur_weights, biases=cur_biases)
    print(f"[+] Saved updated model to: {out_path}")

    # Also update model_data.js and model.json for instant web reflection
    export_web_assets(cur_weights, cur_biases)

    print("\n[SUCCESS] Your custom-labeled handwriting data has been trained into the model!")
    print("Refresh your browser to see the updated model in action.")


if __name__ == "__main__":
    main()
