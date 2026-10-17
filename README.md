# ColdLogic — Handwritten Digit Classifier & Active Learning Pipeline

A 100% static, client-side Machine Learning web application for handwritten digit classification (MNIST), designed specifically to run seamlessly on **GitHub Pages** with **zero backend**, **no Firebase**, and **no server required**.

Includes real-time camera capture, drawing canvas, active human data labeling, and in-browser or Python model retraining.

---

## 🌟 Highlights

- **100% Client-Side Inference**: Runs directly in the user's browser in `< 2ms` with zero API costs, zero external network requests, and complete privacy.
- **Model Architecture**: Uses trained weights and biases matching `models/digit_model.npz`:
  $$\vec{z} = W \cdot \vec{x} + \vec{b}$$
  $$\text{Predicted Class} = \arg\max(\vec{z})$$
  $$\text{Softmax: } P(i) = \frac{e^{z_i - \max(z)}}{\sum_{k=0}^9 e^{z_k - \max(z)}}$$
- **Multi-Modal Input with Live Camera**:
  - **Camera with Smart Reticle**: Point your phone or webcam at paper digits; smart reticle crops the central area to eliminate desk clutter.
  - **File Upload & Drag-and-Drop**: PNG, JPG, JPEG, WEBP, or paste directly from clipboard (`Ctrl+V`).
  - **Touch Drawing Canvas**: Smooth finger drawing on mobile screens with scroll lock (`touch-action: none`).
  - **MNIST Sample Gallery**: Instant test samples for digits 0–9.
- **Human Data Labeling & Active Learning**:
  - **1-Tap Confirm / Correct**: Confirm predictions with one tap or choose the true digit (0–9).
  - **Dataset Persistence**: Labeled samples (`[0.0, 1.0]` normalized 784 vectors + labels) are saved in browser `localStorage`.
  - **Export Dataset**: Download your labeled samples as `mnist_user_dataset.json`.
  - **In-Browser Fine-Tuning**: Click **"Train Now"** to execute Softmax Cross-Entropy gradient descent directly in JavaScript and watch the model adapt in real time!
  - **Python Retraining**: Run `python train_user_data.py` to retrain `models/digit_model_updated.npz` and auto-update `model_data.js` & `model.json`.
- **Mobile-First Responsive Design**:
  - Touch-optimized 44px tap targets.
  - 5&times;2 digit selection grid on small phone screens.
  - Collapsible preprocessing calibration panel.
  - Horizontal snap-scrolling carousel for the 6-stage mathematical visualizer.
  - High-contrast obsidian & graphite palette with no cheesy neon AI blur or blue-purple gradients.

---

## 🚀 How to Host on GitHub Pages (Step-by-Step)

Because this web app requires **no backend or database**, you can host it for free on GitHub Pages in under two minutes:

### Step 1: Commit and Push
```powershell
git add .
git commit -m "Update mobile layout, camera, and active learning data labeling"
git push origin main
```

### Step 2: Enable GitHub Pages
1. Go to your repository on GitHub.
2. Click **Settings** $\rightarrow$ **Pages** (in the left sidebar).
3. Under **Build and deployment > Branch**, select `main` and folder `/ (root)`.
4. Click **Save**.

Your classifier will be live worldwide at:
```
https://YOUR_USERNAME.github.io/YOUR_REPOSITORY_NAME/
```

---

## 🧠 Retraining with Collected User Data

When users label digits on the website, they can download `mnist_user_dataset.json`. To retrain the Python model with this data:

1. Place the downloaded `mnist_user_dataset.json` in this directory.
2. Run:
   ```powershell
   python train_user_data.py
   ```
3. The script will:
   - Load baseline weights from `models/digit_model.npz`.
   - Run gradient descent on your labeled handwriting samples.
   - Save the updated model to `models/digit_model_updated.npz`.
   - Automatically update `model_data.js` and `model.json` for the web app!
