# ColdLogic AI — Handwritten Digit Recognition Web Pipeline

A 100% static, client-side Machine Learning web application for handwritten digit classification (MNIST), designed specifically to run seamlessly on **GitHub Pages** with **zero backend**, **no Firebase**, and **no server required**.

All existing project files (`new.py`, `image.py`, `main.py`, `models/digit_model.npz`) remain completely untouched and intact.

---

## 🌟 Highlights

- **100% Client-Side Inference**: Runs directly in the user's browser in `< 2ms` with zero API costs, zero data sent to external servers, and complete privacy.
- **Identical Model Logic**: Uses the exact model architecture and trained parameters from `models/digit_model.npz`:
  $$\vec{z} = W \cdot \vec{x} + \vec{b}$$
  $$\text{Predicted Class} = \arg\max(\vec{z})$$
  $$\text{Softmax: } P(i) = \frac{e^{z_i - \max(z)}}{\sum_{k=0}^9 e^{z_k - \max(z)}}$$
- **Full Visual Image Pipeline**:
  - **Stage 1 (Raw Input)**: Upload via file picker, drag-and-drop, or clipboard paste (`Ctrl+V`). Also includes a live drawing canvas and a 0–9 sample gallery.
  - **Stage 2 (Grayscale & Auto-Inversion)**: Converts RGB image to luminance ($0.299R + 0.587G + 0.114B$) and automatically detects light/dark paper backgrounds.
  - **Stage 3 (28×28 MNIST Standard Canvas)**: Extracts ink bounding box and centers digit in a 20×20 box within the 28×28 frame.
  - **Stage 4 (784 Normalized Feature Vector)**: Normalizes pixel intensities to range $[0.0, 1.0]$.
  - **Stage 5 (Linear Layer Dot Product)**: Matrix multiplication using the 10×784 weight matrix and 10 biases.
  - **Stage 6 (Argmax & Softmax Probability Distribution)**: Live prediction badge, confidence percentage, top-3 candidates, and animated probability bars for digits 0–9.
- **Glassmorphic Tech UI**: Dark mode palette with glowing neon accents, responsive layout for desktop and mobile, and interactive logits inspector.

---

## 🚀 How to Host on GitHub Pages (Step-by-Step)

Because this web app requires **no backend or database**, you can host it for free on GitHub Pages in under two minutes:

### Step 1: Initialize Git and Commit
Open PowerShell or your terminal in this project folder:
```powershell
git init
git add .
git commit -m "Add MNIST digit recognizer web interface"
git branch -M main
```

### Step 2: Push to Your GitHub Repository
Create a new public repository on [GitHub](https://github.com/new), then connect and push:
```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
git push -u origin main
```

### Step 3: Enable GitHub Pages
1. Go to your repository on GitHub.
2. Click **Settings** (top tab) $\rightarrow$ **Pages** (in the left sidebar).
3. Under **Build and deployment > Source**, ensure **Deploy from a branch** is selected.
4. Under **Branch**, select `main` and folder `/ (root)`.
5. Click **Save**.

That's it! Within 1–2 minutes, your live web app will be accessible worldwide at:
```
https://YOUR_USERNAME.github.io/YOUR_REPOSITORY_NAME/
```

---

## 💻 Running Locally

You can test the application locally in any of the following ways:

### Option A: Double-Click
Simply double-click [`index.html`](index.html) in Windows File Explorer. Because weights are bundled into [`model_data.js`](model_data.js), it runs even without a web server!

### Option B: Built-in Python Server
```powershell
python -m http.server 8080
```
Then navigate to: `http://localhost:8080`

---

## 📁 Project Structure

```text
├── index.html              # Main HTML5 web interface & visual pipeline
├── styles.css              # Modern glassmorphism dark-mode stylesheet
├── app.js                  # Pipeline processing, canvas drawing, & ML inference logic
├── model_data.js           # Exact weights & biases exported from digit_model.npz (7,850 floats)
├── model.json              # Standard JSON format model weights
├── samples.js              # Pre-loaded MNIST test digits (0-9) for 1-click evaluation
├── .gitignore              # Ignores large virtual environments and training caches
│
├── new.py                  # Original model training and forward-pass logic (untouched)
├── image.py                # Original image preprocessing script (untouched)
├── main.py                 # Original CLI interface (untouched)
├── models/
│   ├── digit_model.npz     # Trained weights (10x784) and biases (10) (untouched)
│   └── digit_model_updated.npz
└── data/
    └── mnist.npz           # MNIST dataset archive (untouched)
```

---

## 🧠 Model Specifications

| Parameter | Value |
| :--- | :--- |
| **Input Shape** | $28 \times 28 = 784$ grayscale pixels |
| **Input Normalization** | $X_i \in [0.0, 1.0]$ |
| **Model Type** | Softmax Linear Regression |
| **Weight Tensor** | $10 \times 784$ (7,840 parameters) |
| **Bias Vector** | $10$ parameters |
| **Total Parameters** | 7,850 parameters |
| **Inference Time** | $\approx 1 - 2\text{ ms}$ (Client-Side JavaScript) |
