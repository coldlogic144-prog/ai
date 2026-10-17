"""
ColdLogic AI - Unified Handwritten Digit Classifier & Retraining Engine
1. Combines standard MNIST dataset with user-labeled custom samples.
2. Provides automatic web data collection and model retraining APIs.
3. Can run directly from terminal (python train.py) or as a live web server (python train.py --serve).
"""

import os
import sys
import json
import glob
import time
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
USER_DATA_FILE = DATA_DIR / "user_samples.json"
MNIST_FILE = DATA_DIR / "mnist.npz"
MODEL_NPZ = MODELS_DIR / "digit_model.npz"
UPDATED_MODEL_NPZ = MODELS_DIR / "digit_model_updated.npz"
MODEL_JSON = PROJECT_ROOT / "model.json"
MODEL_DATA_JS = PROJECT_ROOT / "model_data.js"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# =========================================================================
# 1. DATASET MANAGEMENT
# =========================================================================

def load_existing_mnist(sample_limit=5000):
    """
    Loads baseline MNIST training and test datasets.
    Uses sample_limit for quick, balanced fine-tuning alongside user data.
    """
    if not MNIST_FILE.exists():
        print(f"[!] MNIST dataset not found at {MNIST_FILE}. Downloading...")
        import urllib.request
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        urllib.request.urlretrieve(url, MNIST_FILE)
        print(f"[+] Downloaded MNIST dataset to {MNIST_FILE}")

    data = np.load(MNIST_FILE)
    x_train_raw = data["x_train"]
    y_train_raw = data["y_train"]
    x_test_raw = data["x_test"]
    y_test_raw = data["y_test"]

    # Flatten and normalize to [0.0, 1.0]
    x_train = (x_train_raw.reshape(-1, 784) / 255.0).astype(np.float32)
    y_train = y_train_raw.astype(np.int64)

    x_test = (x_test_raw.reshape(-1, 784) / 255.0).astype(np.float32)
    y_test = y_test_raw.astype(np.int64)

    if sample_limit and sample_limit < len(x_train):
        # Balanced sampling across all 10 digits
        indices = []
        per_digit = sample_limit // 10
        for digit in range(10):
            digit_indices = np.where(y_train == digit)[0]
            indices.extend(digit_indices[:per_digit])
        indices = np.array(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

    return x_train, y_train, x_test, y_test


def load_user_samples():
    """
    Loads user-labeled samples automatically collected from the webpage.
    Searches data/user_samples.json, then any exported mnist_user_dataset*.json as fallback.
    """
    target = None
    if USER_DATA_FILE.exists():
        target = USER_DATA_FILE
    else:
        # Check current folder or Downloads for exported files
        candidates = sorted(glob.glob("mnist_user_dataset*.json"), reverse=True)
        if not candidates:
            downloads = Path.home() / "Downloads"
            candidates = sorted(glob.glob(str(downloads / "mnist_user_dataset*.json")), reverse=True)
        if candidates:
            target = Path(candidates[0])

    if not target or not target.exists():
        return np.empty((0, 784), dtype=np.float32), np.empty((0,), dtype=np.int64), []

    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("samples", [])
        if not samples and isinstance(data, list):
            samples = data

        x_list = []
        y_list = []
        for s in samples:
            if "vector" in s and "label" in s:
                x_list.append(s["vector"])
                y_list.append(int(s["label"]))

        if not x_list:
            return np.empty((0, 784), dtype=np.float32), np.empty((0,), dtype=np.int64), []

        return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.int64), samples
    except Exception as e:
        print(f"[!] Error loading user samples from {target}: {e}")
        return np.empty((0, 784), dtype=np.float32), np.empty((0,), dtype=np.int64), []


def append_user_sample(sample_dict):
    """
    Automatically saves a user-labeled sample into data/user_samples.json.
    """
    DATA_DIR.mkdir(exist_ok=True)
    existing_samples = []

    if USER_DATA_FILE.exists():
        try:
            with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing_samples = data.get("samples", [])
        except Exception:
            existing_samples = []

    # Assign ID & timestamp if not present
    if "id" not in sample_dict:
        sample_dict["id"] = f"sample_{int(time.time()*1000)}"
    if "timestamp" not in sample_dict:
        sample_dict["timestamp"] = int(time.time()*1000)

    existing_samples.append(sample_dict)

    payload = {
        "version": 1,
        "description": "ColdLogic Collected User Labeled Digits",
        "lastUpdated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "totalSamples": len(existing_samples),
        "samples": existing_samples
    }

    with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return len(existing_samples)


# =========================================================================
# 2. MACHINE LEARNING TRAINING ROUTINE
# =========================================================================

def load_or_init_model():
    """Loads existing weights and biases or initializes new parameters."""
    if MODEL_NPZ.exists():
        saved = np.load(MODEL_NPZ)
        weights = saved["weights"].astype(np.float32)
        biases = saved["biases"].astype(np.float32)
        print(f"[+] Loaded baseline model from {MODEL_NPZ.name}")
    else:
        print("[*] No existing model found. Initializing random weights.")
        weights = (np.random.randn(10, 784) * 0.01).astype(np.float32)
        biases = np.zeros(10, dtype=np.float32)
    return weights, biases


def evaluate_accuracy(weights, biases, x, y):
    """Calculates classification accuracy percentage."""
    if len(x) == 0:
        return 0.0
    logits = np.dot(x, weights.T) + biases
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y) * 100.0)


def train_combined_model(epochs=25, learning_rate=0.08, user_oversample=30, mnist_sample_count=4000):
    """
    Trains the ML model using BOTH:
    1. The baseline MNIST dataset (data/mnist.npz)
    2. The user-collected labeled data (data/user_samples.json)

    User samples are strategically oversampled so the model actively adapts
    to the user's personal digit handwriting without forgetting baseline digits.
    """
    print("=" * 60)
    print("   ColdLogic AI - Training Combined Neural Model")
    print("=" * 60)

    # 1. Load baseline MNIST data
    x_mnist, y_mnist, x_test, y_test = load_existing_mnist(sample_limit=mnist_sample_count)
    print(f"[+] Loaded {len(x_mnist)} MNIST training images and {len(x_test)} validation images.")

    # 2. Load user samples
    x_user, y_user, user_samples_raw = load_user_samples()
    user_count = len(x_user)
    print(f"[+] Loaded {user_count} user-labeled samples from webpage.")

    # 3. Load baseline model
    weights, biases = load_or_init_model()

    # Pre-training evaluation
    initial_mnist_acc = evaluate_accuracy(weights, biases, x_test, y_test)
    initial_user_acc = evaluate_accuracy(weights, biases, x_user, y_user) if user_count > 0 else 0.0

    print(f"[*] Pre-training Accuracy on MNIST Test Set: {initial_mnist_acc:.2f}%")
    if user_count > 0:
        print(f"[*] Pre-training Accuracy on User Samples:    {initial_user_acc:.2f}%")

    # 4. Combine Datasets with Balanced Oversampling
    if user_count > 0:
        # Repeat user samples so their gradient signal adapts the model
        repeats = max(1, user_oversample)
        x_user_repeated = np.repeat(x_user, repeats, axis=0)
        y_user_repeated = np.repeat(y_user, repeats, axis=0)

        x_combined = np.vstack([x_mnist, x_user_repeated])
        y_combined = np.concatenate([y_mnist, y_user_repeated])
        print(f"[+] Formed combined training set: {len(x_mnist)} MNIST + {len(x_user_repeated)} oversampled user samples = {len(x_combined)} total.")
    else:
        x_combined = x_mnist
        y_combined = y_mnist
        print("[*] No user samples available yet. Training on baseline MNIST dataset.")

    # Shuffle training set
    shuffle_idx = np.random.permutation(len(x_combined))
    x_combined = x_combined[shuffle_idx]
    y_combined = y_combined[shuffle_idx]

    # 5. Gradient Descent (Softmax Cross-Entropy)
    N = len(x_combined)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Forward pass: z = X @ W.T + b
        z = np.dot(x_combined, weights.T) + biases

        # Softmax probabilities (numerically stable)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        # Cross-entropy gradient: dz = (probs - one_hot) / N
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(N), y_combined] = 1.0
        dz = (probs - one_hot) / N

        # Parameter gradients
        dw = np.dot(dz.T, x_combined)  # (10, 784)
        db = np.sum(dz, axis=0)         # (10,)

        # Update weights and biases
        weights -= learning_rate * dw
        biases -= learning_rate * db

        if epoch % 5 == 0 or epoch == epochs:
            epoch_acc = float(np.mean(np.argmax(z, axis=1) == y_combined) * 100.0)
            print(f"    Epoch {epoch:2d}/{epochs:2d} - Train Acc: {epoch_acc:.2f}%")

    train_time = round(time.time() - t0, 3)
    print(f"[+] Training completed in {train_time} seconds.")

    # 6. Post-training Evaluation
    final_mnist_acc = evaluate_accuracy(weights, biases, x_test, y_test)
    final_user_acc = evaluate_accuracy(weights, biases, x_user, y_user) if user_count > 0 else 0.0

    print("-" * 60)
    print(f"[+] Final MNIST Test Accuracy: {final_mnist_acc:.2f}% ({final_mnist_acc - initial_mnist_acc:+.2f}%)")
    if user_count > 0:
        print(f"[+] Final User Sample Accuracy: {final_user_acc:.2f}% ({final_user_acc - initial_user_acc:+.2f}%)")
    print("-" * 60)

    # 7. Save Models and Web Assets
    np.savez(UPDATED_MODEL_NPZ, weights=weights, biases=biases)
    np.savez(MODEL_NPZ, weights=weights, biases=biases)
    print(f"[+] Saved updated NumPy weights to: {UPDATED_MODEL_NPZ.name} & {MODEL_NPZ.name}")

    export_payload = {
        "weights": weights.tolist(),
        "biases": biases.tolist()
    }

    with open(MODEL_JSON, "w", encoding="utf-8") as f:
        json.dump(export_payload, f)
    print(f"[+] Updated {MODEL_JSON.name}")

    with open(MODEL_DATA_JS, "w", encoding="utf-8") as f:
        f.write(f"// ColdLogic - MNIST Digit Recognizer Exported Weights & Biases\nwindow.MODEL_DATA = {json.dumps(export_payload)};\n")
    print(f"[+] Updated {MODEL_DATA_JS.name} (static browser loader)")

    return {
        "success": True,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "train_time_sec": train_time,
        "mnist_count": len(x_mnist),
        "user_count": user_count,
        "initial_mnist_acc": round(initial_mnist_acc, 2),
        "final_mnist_acc": round(final_mnist_acc, 2),
        "initial_user_acc": round(initial_user_acc, 2),
        "final_user_acc": round(final_user_acc, 2),
        "weights": export_payload["weights"],
        "biases": export_payload["biases"]
    }


# =========================================================================
# 3. EMBEDDED WEB SERVER WITH AUTOMATIC DATA COLLECTION & TRAINING APIS
# =========================================================================

class ColdLogicServerHandler(SimpleHTTPRequestHandler):
    """
    Serves static frontend assets while providing REST endpoints for:
    - POST /api/save_sample: Automatically appends labeled digit data to data/user_samples.json
    - POST /api/train: Executes train_combined_model() and returns live stats + new weights
    - GET /api/status: Returns current dataset statistics
    """

    def do_GET(self):
        if self.path == "/api/status":
            _, _, samples = load_user_samples()
            self._send_json({
                "status": "ready",
                "user_samples_count": len(samples),
                "has_mnist": MNIST_FILE.exists(),
                "model_exists": MODEL_NPZ.exists()
            })
            return

        # Default static file handler
        return super().do_GET()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/save_sample":
            self._handle_save_sample()
        elif self.path == "/api/train":
            self._handle_train()
        else:
            self.send_error(404, "Endpoint not found")

    def _read_body_json(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return None
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _handle_save_sample(self):
        """Automatically saves a sample sent from the webpage with proper labels."""
        try:
            sample_data = self._read_body_json()
            if not sample_data or "vector" not in sample_data or "label" not in sample_data:
                self._send_json({"success": False, "error": "Invalid sample payload"}, status=400)
                return

            label = int(sample_data["label"])
            total_saved = append_user_sample(sample_data)
            print(f"[API] Automatically collected sample: Label={label} | Total on disk: {total_saved}")

            self._send_json({
                "success": True,
                "label": label,
                "total_samples": total_saved,
                "message": f"Sample labeled as {label} saved to {USER_DATA_FILE.name}"
            })
        except Exception as e:
            print(f"[!] Error saving sample: {e}")
            self._send_json({"success": False, "error": str(e)}, status=500)

    def _handle_train(self):
        """Runs the combined training routine when triggered from the webpage."""
        try:
            req_data = self._read_body_json() or {}
            epochs = int(req_data.get("epochs", 25))
            lr = float(req_data.get("learning_rate", 0.08))

            print("\n[API] Webpage triggered Python model training!")
            result = train_combined_model(epochs=epochs, learning_rate=lr)

            self._send_json(result)
        except Exception as e:
            print(f"[!] Error during training: {e}")
            self._send_json({"success": False, "error": str(e)}, status=500)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()


def run_server(port=8080):
    """Starts the ColdLogic server with static files + API endpoints."""
    server_address = ("", port)
    httpd = HTTPServer(server_address, ColdLogicServerHandler)
    print("=" * 60)
    print(f"  ColdLogic AI Server Running at: http://localhost:{port}/")
    print("  - Automatic Web Data Collection: POST /api/save_sample")
    print("  - Web-Triggered Python Training: POST /api/train")
    print("  - Static Frontend: index.html, styles.css, app.js")
    print("=" * 60)
    print("Press Ctrl+C to stop the server.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")


# =========================================================================
# 4. CLI ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ColdLogic AI - Train digit model on MNIST + user data or run live web server"
    )
    parser.add_argument(
        "--serve", "-s",
        action="store_true",
        help="Run live web server with automated data collection and retraining APIs"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Server port (default: 8080)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=25,
        help="Training epochs (default: 25)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.08,
        help="Learning rate (default: 0.08)"
    )

    args = parser.parse_args()

    if args.serve:
        run_server(port=args.port)
    else:
        # Train directly from CLI
        train_combined_model(epochs=args.epochs, learning_rate=args.lr)
