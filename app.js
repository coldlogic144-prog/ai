/**
 * ColdLogic - Handwritten Digit Classifier
 * 100% Client-Side Machine Learning Pipeline
 * With Human Data Labeling & In-Browser Fine-Tuning
 */

(function () {
  'use strict';

  const STORAGE_KEY = 'coldlogic_mnist_dataset_v1';

  // --- Global Application State ---
  const state = {
    model: null,
    currentSource: null,
    currentSourceName: 'Sample 0',
    activeTab: 'upload',
    settings: {
      alignMode: 'smart',
      invertMode: 'auto',
      contrast: 1.4,
    },
    isDrawing: false,
    lastX: 0,
    lastY: 0,
    brushSize: 22,
    debounceTimer: null,
    camera: {
      stream: null,
      facingMode: 'environment',
      isActive: false
    },
    // Active Learning & Data Labeling
    currentInputVector: null,
    currentWinner: null,
    currentConfidence: 0,
    dataset: []
  };

  // --- DOM Elements ---
  const elements = {
    // Nav & Badges
    modelStatusPill: document.getElementById('modelStatusPill'),
    modelStatusText: document.getElementById('modelStatusText'),
    deployGuideBtn: document.getElementById('deployGuideBtn'),
    deployModal: document.getElementById('deployModal'),
    closeDeployModalBtn: document.getElementById('closeDeployModalBtn'),
    gotItModalBtn: document.getElementById('gotItModalBtn'),

    // Hero Dataset Indicator
    heroDatasetCount: document.getElementById('heroDatasetCount'),

    // Tabs
    tabUploadBtn: document.getElementById('tabUploadBtn'),
    tabCameraBtn: document.getElementById('tabCameraBtn'),
    tabDrawBtn: document.getElementById('tabDrawBtn'),
    tabSamplesBtn: document.getElementById('tabSamplesBtn'),
    paneUpload: document.getElementById('paneUpload'),
    paneCamera: document.getElementById('paneCamera'),
    paneDraw: document.getElementById('paneDraw'),
    paneSamples: document.getElementById('paneSamples'),

    // Upload Pane
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('fileInput'),
    browseFileBtn: document.getElementById('browseFileBtn'),
    quickCameraBtn: document.getElementById('quickCameraBtn'),

    // Camera Pane
    cameraViewfinder: document.getElementById('cameraViewfinder'),
    cameraVideo: document.getElementById('cameraVideo'),
    cameraFlash: document.getElementById('cameraFlash'),
    cameraIdleOverlay: document.getElementById('cameraIdleOverlay'),
    cameraStatusText: document.getElementById('cameraStatusText'),
    cameraStatusSub: document.getElementById('cameraStatusSub'),
    startCameraBtn: document.getElementById('startCameraBtn'),
    deviceCameraBtn: document.getElementById('deviceCameraBtn'),
    nativeCameraInput: document.getElementById('nativeCameraInput'),
    capturePhotoBtn: document.getElementById('capturePhotoBtn'),
    flipCameraBtn: document.getElementById('flipCameraBtn'),
    nativeShutterToolbarBtn: document.getElementById('nativeShutterToolbarBtn'),
    stopCameraBtn: document.getElementById('stopCameraBtn'),

    // Draw Pane
    drawingCanvas: document.getElementById('drawingCanvas'),
    brushSize: document.getElementById('brushSize'),
    brushSizeVal: document.getElementById('brushSizeVal'),
    clearCanvasBtn: document.getElementById('clearCanvasBtn'),

    // Samples Pane
    samplesGrid: document.getElementById('samplesGrid'),

    // Settings
    alignModeSelect: document.getElementById('alignModeSelect'),
    invertSelect: document.getElementById('invertSelect'),
    contrastRange: document.getElementById('contrastRange'),
    contrastVal: document.getElementById('contrastVal'),
    resetSettingsBtn: document.getElementById('resetSettingsBtn'),

    // Prediction Output
    resultsCard: document.getElementById('resultsCard'),
    heroPredictionCard: document.getElementById('heroPredictionCard'),
    predictedNumber: document.getElementById('predictedNumber'),
    confidenceValue: document.getElementById('confidenceValue'),
    topThreeChips: document.getElementById('topThreeChips'),
    inferenceBadge: document.getElementById('inferenceBadge'),
    distMaxNote: document.getElementById('distMaxNote'),
    probabilityBars: document.getElementById('probabilityBars'),
    logitsTableBody: document.getElementById('logitsTableBody'),

    // Data Labeling & Dataset Management
    labelingCard: document.getElementById('labelingCard'),
    confirmCorrectBtn: document.getElementById('confirmCorrectBtn'),
    quickDigitText: document.getElementById('quickDigitText'),
    digitChipsGrid: document.getElementById('digitChipsGrid'),
    labelStatusToast: document.getElementById('labelStatusToast'),
    labelStatusMsg: document.getElementById('labelStatusMsg'),
    datasetCounterBadge: document.getElementById('datasetCounterBadge'),
    datasetCountLabel: document.getElementById('datasetCountLabel'),
    exportDatasetBtn: document.getElementById('exportDatasetBtn'),
    btnDatasetCount: document.getElementById('btnDatasetCount'),
    trainBrowserBtn: document.getElementById('trainBrowserBtn'),
    clearDatasetBtn: document.getElementById('clearDatasetBtn'),

    // Pipeline Stages
    rawPreviewCanvas: document.getElementById('rawPreviewCanvas'),
    rawSizeVal: document.getElementById('rawSizeVal'),
    rawSourceVal: document.getElementById('rawSourceVal'),
    grayCanvas: document.getElementById('grayCanvas'),
    invertStateVal: document.getElementById('invertStateVal'),
    mnistCanvas: document.getElementById('mnistCanvas'),
    stageAlignVal: document.getElementById('stageAlignVal'),
    vectorHeatmapCanvas: document.getElementById('vectorHeatmapCanvas'),
    nonZeroCount: document.getElementById('nonZeroCount'),
    miniPredictedNum: document.getElementById('miniPredictedNum'),
    miniPredictedConf: document.getElementById('miniPredictedConf'),
    stageWinnerVal: document.getElementById('stageWinnerVal')
  };

  // --- Initialize Application ---
  async function initApp() {
    loadSavedDataset();
    initModel();
    initTabs();
    initUpload();
    initCamera();
    initDrawingCanvas();
    initSamples();
    initSettings();
    initLabelingControls();
    initModal();
    initPasteHandler();

    // Load initial sample digit (0)
    setTimeout(() => {
      loadInitialDigit();
    }, 100);
  }

  // --- Dataset Persistence ---
  function loadSavedDataset() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        state.dataset = JSON.parse(saved);
      }
    } catch (e) {
      console.warn('Could not read from localStorage:', e);
      state.dataset = [];
    }
    updateDatasetUI();
  }

  function saveDatasetToStorage() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state.dataset));
    } catch (e) {
      console.warn('Could not save to localStorage (quota or disabled):', e);
    }
    updateDatasetUI();
  }

  function updateDatasetUI() {
    const count = state.dataset.length;
    const label = `${count} ${count === 1 ? 'sample' : 'samples'} saved`;
    if (elements.datasetCountLabel) elements.datasetCountLabel.textContent = label;
    if (elements.heroDatasetCount) elements.heroDatasetCount.textContent = count;
    if (elements.btnDatasetCount) elements.btnDatasetCount.textContent = count;

    const hasData = count > 0;
    if (elements.exportDatasetBtn) elements.exportDatasetBtn.disabled = !hasData;
    if (elements.trainBrowserBtn) elements.trainBrowserBtn.disabled = !hasData;
    if (elements.clearDatasetBtn) elements.clearDatasetBtn.disabled = !hasData;
  }

  // --- Model Loader ---
  async function initModel() {
    try {
      if (window.MODEL_DATA && window.MODEL_DATA.weights && window.MODEL_DATA.biases) {
        // Deep copy weights and biases so user in-browser training does not mutate raw file globals
        state.model = {
          weights: window.MODEL_DATA.weights.map(row => Array.from(row)),
          biases: Array.from(window.MODEL_DATA.biases)
        };
        onModelLoaded();
        return;
      }

      elements.modelStatusText.textContent = 'Fetching model.json...';
      const response = await fetch('model.json');
      if (!response.ok) throw new Error('Failed to load model.json');
      const loaded = await response.json();
      state.model = {
        weights: loaded.weights.map(row => Array.from(row)),
        biases: Array.from(loaded.biases)
      };
      onModelLoaded();
    } catch (err) {
      console.error('Error loading model weights:', err);
      elements.modelStatusPill.classList.remove('status-ready');
      elements.modelStatusPill.style.borderColor = 'rgba(239, 68, 68, 0.4)';
      elements.modelStatusPill.style.color = '#ef4444';
      elements.modelStatusText.textContent = 'Load Failed';
    }
  }

  function onModelLoaded() {
    elements.modelStatusPill.classList.add('status-ready');
    elements.modelStatusText.textContent = `Model Ready`;
  }

  // --- Tab Navigation ---
  function initTabs() {
    const tabs = [
      { btn: elements.tabUploadBtn, pane: elements.paneUpload, id: 'upload' },
      { btn: elements.tabCameraBtn, pane: elements.paneCamera, id: 'camera' },
      { btn: elements.tabDrawBtn, pane: elements.paneDraw, id: 'draw' },
      { btn: elements.tabSamplesBtn, pane: elements.paneSamples, id: 'samples' }
    ];

    tabs.forEach(tab => {
      tab.btn.addEventListener('click', () => {
        const prevTab = state.activeTab;

        tabs.forEach(t => {
          t.btn.classList.remove('active');
          t.btn.setAttribute('aria-selected', 'false');
          t.pane.classList.remove('active');
        });

        tab.btn.classList.add('active');
        tab.btn.setAttribute('aria-selected', 'true');
        tab.pane.classList.add('active');
        state.activeTab = tab.id;

        // Auto stop camera when leaving camera tab to save battery
        if (prevTab === 'camera' && tab.id !== 'camera') {
          stopCamera();
        }

        // Auto start camera when switching to camera tab
        if (tab.id === 'camera' && !state.camera.isActive) {
          startCamera();
        }

        if (tab.id === 'draw') {
          processSource(elements.drawingCanvas, 'Live Drawing');
        }
      });
    });
  }

  // --- Upload & Drag & Drop ---
  function initUpload() {
    elements.browseFileBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      elements.fileInput.click();
    });

    if (elements.quickCameraBtn) {
      elements.quickCameraBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        elements.tabCameraBtn.click();
      });
    }

    elements.dropzone.addEventListener('click', () => {
      elements.fileInput.click();
    });

    elements.fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
      }
    });

    // Drag and drop events
    ['dragenter', 'dragover'].forEach(eventName => {
      elements.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        elements.dropzone.classList.add('dragover');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      elements.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        elements.dropzone.classList.remove('dragover');
      }, false);
    });

    elements.dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files[0]) {
        handleFile(dt.files[0]);
      }
    });
  }

  // Clipboard paste support (Ctrl+V anywhere on page)
  function initPasteHandler() {
    window.addEventListener('paste', (e) => {
      if (e.clipboardData && e.clipboardData.items) {
        for (let i = 0; i < e.clipboardData.items.length; i++) {
          const item = e.clipboardData.items[i];
          if (item.type.indexOf('image') !== -1) {
            const blob = item.getAsFile();
            handleFile(blob, 'Pasted Image');
            elements.tabUploadBtn.click();
            break;
          }
        }
      }
    });
  }

  function handleFile(file, customName) {
    if (!file || !file.type.startsWith('image/')) {
      alert('Please upload a valid image file (PNG, JPG, JPEG, WEBP).');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        processSource(img, customName || file.name);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  // --- Live Camera Controller ---
  function initCamera() {
    elements.startCameraBtn.addEventListener('click', () => {
      startCamera();
    });

    elements.stopCameraBtn.addEventListener('click', () => {
      stopCamera();
    });

    elements.flipCameraBtn.addEventListener('click', () => {
      toggleCameraFlip();
    });

    elements.capturePhotoBtn.addEventListener('click', () => {
      captureCameraPhoto();
    });

    // Native mobile camera fallback triggers
    const triggerNativeShutter = () => {
      elements.nativeCameraInput.click();
    };

    if (elements.deviceCameraBtn) {
      elements.deviceCameraBtn.addEventListener('click', triggerNativeShutter);
    }
    if (elements.nativeShutterToolbarBtn) {
      elements.nativeShutterToolbarBtn.addEventListener('click', triggerNativeShutter);
    }

    elements.nativeCameraInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0], 'Camera Photo');
      }
    });
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showCameraError('Camera Not Supported', 'Your browser does not support camera streaming. Use Phone Shutter instead.');
      return;
    }

    elements.cameraStatusText.textContent = 'Connecting...';
    elements.cameraStatusSub.textContent = 'Please allow camera permission if requested.';

    if (state.camera.stream) {
      state.camera.stream.getTracks().forEach(track => track.stop());
      state.camera.stream = null;
    }

    try {
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: state.camera.facingMode,
            width: { ideal: 1280 },
            height: { ideal: 720 }
          },
          audio: false
        });
      } catch (err) {
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false
        });
      }

      state.camera.stream = stream;
      state.camera.isActive = true;
      elements.cameraVideo.srcObject = stream;

      if (state.camera.facingMode === 'user') {
        elements.cameraVideo.style.transform = 'scaleX(-1)';
      } else {
        elements.cameraVideo.style.transform = 'scaleX(1)';
      }

      elements.cameraVideo.onloadedmetadata = () => {
        elements.cameraVideo.play();
        elements.cameraIdleOverlay.classList.add('hidden');
        elements.capturePhotoBtn.disabled = false;
      };
    } catch (err) {
      console.warn('Camera access denied:', err);
      showCameraError('Camera Denied', 'Permission was denied or camera is in use. You can use the Phone Shutter or upload an image.');
    }
  }

  function stopCamera() {
    if (state.camera.stream) {
      state.camera.stream.getTracks().forEach(track => track.stop());
      state.camera.stream = null;
    }
    state.camera.isActive = false;
    elements.cameraVideo.srcObject = null;
    elements.capturePhotoBtn.disabled = true;

    elements.cameraStatusText.textContent = 'Camera Inactive';
    elements.cameraStatusSub.textContent = 'Start live camera or use your phone\'s native camera shutter.';
    elements.cameraIdleOverlay.classList.remove('hidden');
  }

  function toggleCameraFlip() {
    state.camera.facingMode = state.camera.facingMode === 'environment' ? 'user' : 'environment';
    if (state.camera.isActive) {
      startCamera();
    }
  }

  function showCameraError(title, subtitle) {
    elements.cameraStatusText.textContent = title;
    elements.cameraStatusSub.textContent = subtitle;
    elements.cameraIdleOverlay.classList.remove('hidden');
    elements.capturePhotoBtn.disabled = true;
  }

  function captureCameraPhoto() {
    if (!state.camera.isActive || !elements.cameraVideo.videoWidth) {
      return;
    }

    if (navigator.vibrate) navigator.vibrate(40);

    // Visual shutter flash
    elements.cameraFlash.classList.add('flash-active');
    setTimeout(() => {
      elements.cameraFlash.classList.remove('flash-active');
    }, 300);

    const video = elements.cameraVideo;
    const vw = video.videoWidth;
    const vh = video.videoHeight;

    // Crop to the centered reticle viewfinder area (~62% of the shorter side)
    const cropSize = Math.round(Math.min(vw, vh) * 0.62);
    const cropX = Math.round((vw - cropSize) / 2);
    const cropY = Math.round((vh - cropSize) / 2);

    const canvas = document.createElement('canvas');
    canvas.width = cropSize;
    canvas.height = cropSize;
    const ctx = canvas.getContext('2d');

    if (state.camera.facingMode === 'user') {
      ctx.translate(cropSize, 0);
      ctx.scale(-1, 1);
    }

    ctx.drawImage(video, cropX, cropY, cropSize, cropSize, 0, 0, cropSize, cropSize);

    processSource(canvas, 'Camera Snapshot');
  }

  // --- Interactive Drawing Canvas (With Mobile Touch-Lock) ---
  function initDrawingCanvas() {
    const canvas = elements.drawingCanvas;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    clearCanvas();

    elements.brushSize.addEventListener('input', (e) => {
      state.brushSize = parseInt(e.target.value, 10);
      elements.brushSizeVal.textContent = `${state.brushSize}px`;
    });

    elements.clearCanvasBtn.addEventListener('click', () => {
      clearCanvas();
      processSource(canvas, 'Cleared Canvas');
    });

    function clearCanvas() {
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    function getCoords(e) {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      let clientX = e.clientX;
      let clientY = e.clientY;

      if (e.touches && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
      }

      return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY
      };
    }

    function startDrawing(e) {
      state.isDrawing = true;
      const coords = getCoords(e);
      state.lastX = coords.x;
      state.lastY = coords.y;
      draw(e);
    }

    function draw(e) {
      if (!state.isDrawing) return;
      if (e.cancelable) e.preventDefault();

      const coords = getCoords(e);

      ctx.strokeStyle = '#ffffff';
      ctx.fillStyle = '#ffffff';
      ctx.lineWidth = state.brushSize;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.beginPath();
      ctx.moveTo(state.lastX, state.lastY);
      ctx.lineTo(coords.x, coords.y);
      ctx.stroke();

      state.lastX = coords.x;
      state.lastY = coords.y;

      debouncedProcessDrawing();
    }

    function stopDrawing() {
      if (state.isDrawing) {
        state.isDrawing = false;
        processSource(canvas, 'Live Drawing');
      }
    }

    // Pointer events (modern desktop & touch)
    canvas.addEventListener('pointerdown', startDrawing);
    canvas.addEventListener('pointermove', draw);
    window.addEventListener('pointerup', stopDrawing);
    window.addEventListener('pointercancel', stopDrawing);

    // Explicit touch listeners with passive: false to prevent mobile screen bouncing/scrolling while drawing
    canvas.addEventListener('touchstart', (e) => {
      if (e.cancelable) e.preventDefault();
      startDrawing(e);
    }, { passive: false });

    canvas.addEventListener('touchmove', (e) => {
      if (e.cancelable) e.preventDefault();
      draw(e);
    }, { passive: false });

    canvas.addEventListener('touchend', (e) => {
      stopDrawing();
    }, { passive: false });
  }

  function debouncedProcessDrawing() {
    if (state.debounceTimer) clearTimeout(state.debounceTimer);
    state.debounceTimer = setTimeout(() => {
      processSource(elements.drawingCanvas, 'Live Drawing');
    }, 120);
  }

  // --- Samples Gallery ---
  function initSamples() {
    const grid = elements.samplesGrid;
    grid.innerHTML = '';

    if (!window.SAMPLE_DIGITS) {
      grid.innerHTML = '<p class="text-dim">No pre-loaded samples found.</p>';
      return;
    }

    Object.keys(window.SAMPLE_DIGITS).forEach(key => {
      const sample = window.SAMPLE_DIGITS[key];
      const item = document.createElement('button');
      item.type = 'button';
      item.className = 'sample-item';
      item.id = `sampleBtn_${sample.digit}`;
      item.innerHTML = `
        <img src="${sample.image}" alt="Digit ${sample.digit}" class="sample-thumb" />
        <span class="sample-label">Digit ${sample.digit}</span>
      `;

      item.addEventListener('click', () => {
        document.querySelectorAll('.sample-item').forEach(el => el.classList.remove('active'));
        item.classList.add('active');

        const img = new Image();
        img.onload = () => {
          processSource(img, `Sample ${sample.digit}`);
        };
        img.src = sample.image;
      });

      grid.appendChild(item);
    });
  }

  function loadInitialDigit() {
    if (window.SAMPLE_DIGITS && window.SAMPLE_DIGITS['0']) {
      const sampleBtn = document.getElementById('sampleBtn_0');
      if (sampleBtn) sampleBtn.classList.add('active');

      const img = new Image();
      img.onload = () => {
        processSource(img, 'Sample 0');
      };
      img.src = window.SAMPLE_DIGITS['0'].image;
    }
  }

  // --- Settings Panel ---
  function initSettings() {
    elements.alignModeSelect.addEventListener('change', (e) => {
      state.settings.alignMode = e.target.value;
      reprocessCurrent();
    });

    elements.invertSelect.addEventListener('change', (e) => {
      state.settings.invertMode = e.target.value;
      reprocessCurrent();
    });

    elements.contrastRange.addEventListener('input', (e) => {
      state.settings.contrast = parseFloat(e.target.value);
      elements.contrastVal.textContent = `${state.settings.contrast.toFixed(1)}x`;
      reprocessCurrent();
    });

    elements.resetSettingsBtn.addEventListener('click', () => {
      elements.alignModeSelect.value = 'smart';
      elements.invertSelect.value = 'auto';
      elements.contrastRange.value = '1.4';
      elements.contrastVal.textContent = '1.4x';

      state.settings.alignMode = 'smart';
      state.settings.invertMode = 'auto';
      state.settings.contrast = 1.4;

      reprocessCurrent();
    });
  }

  function reprocessCurrent() {
    if (state.currentSource) {
      processSource(state.currentSource, state.currentSourceName);
    }
  }

  // --- Active Learning: Data Labeling Controls ---
  function initLabelingControls() {
    // 1-Tap Quick Confirmation
    elements.confirmCorrectBtn.addEventListener('click', () => {
      if (state.currentWinner === null) return;
      saveCurrentSample(state.currentWinner);
    });

    // 0-9 Chip Selector Grid
    const chips = elements.digitChipsGrid.querySelectorAll('.digit-chip');
    chips.forEach(chip => {
      chip.addEventListener('click', () => {
        chips.forEach(c => c.classList.remove('active-picked'));
        chip.classList.add('active-picked');

        const label = parseInt(chip.dataset.digit, 10);
        saveCurrentSample(label);
      });
    });

    // Download Dataset (.json)
    elements.exportDatasetBtn.addEventListener('click', exportDatasetAsJson);

    // In-Browser Gradient Descent Fine-Tuning
    elements.trainBrowserBtn.addEventListener('click', trainModelInBrowser);

    // Clear Dataset
    elements.clearDatasetBtn.addEventListener('click', () => {
      if (state.dataset.length === 0) return;
      if (confirm(`Clear all ${state.dataset.length} saved training samples?`)) {
        state.dataset = [];
        saveDatasetToStorage();
        showToast('Dataset cleared.');
      }
    });
  }

  function saveCurrentSample(label) {
    if (!state.currentInputVector) {
      alert('Please provide an input image, photo, or drawing first.');
      return;
    }

    if (navigator.vibrate) navigator.vibrate(30);

    const sample = {
      id: 'sample_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
      label: label,
      predicted: state.currentWinner,
      confidence: Number((state.currentConfidence * 100).toFixed(1)),
      vector: Array.from(state.currentInputVector), // 784 normalized [0.0, 1.0]
      timestamp: Date.now(),
      source: state.currentSourceName
    };

    state.dataset.push(sample);
    saveDatasetToStorage();

    showToast(`✓ Saved sample labeled as Digit ${label}! (Total: ${state.dataset.length})`);
  }

  function showToast(message) {
    elements.labelStatusMsg.textContent = message;
    elements.labelStatusToast.style.display = 'flex';
    clearTimeout(elements.labelStatusToast._timer);
    elements.labelStatusToast._timer = setTimeout(() => {
      elements.labelStatusToast.style.display = 'none';
    }, 4000);
  }

  function exportDatasetAsJson() {
    if (state.dataset.length === 0) {
      alert('No labeled samples to export yet.');
      return;
    }

    const payload = {
      version: 1,
      modelType: 'Softmax Linear Regression',
      format: 'MNIST 784 Normalized Pixels [0.0, 1.0]',
      exportedAt: new Date().toISOString(),
      sampleCount: state.dataset.length,
      samples: state.dataset
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mnist_user_dataset_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // --- In-Browser Gradient Descent Fine-Tuning ---
  // Mirrors new.py's update_weights_and_biases directly in JavaScript
  function trainModelInBrowser() {
    if (state.dataset.length === 0) {
      alert('Save at least 1 labeled sample before training.');
      return;
    }

    const lr = 0.08;
    const epochs = 5;
    const samples = state.dataset;

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let s = 0; s < samples.length; s++) {
        const item = samples[s];
        const x = item.vector; // 784
        const y = item.label;  // 0-9

        // 1. Forward Pass Logits: z = W · x + b
        const z = new Float32Array(10);
        for (let c = 0; c < 10; c++) {
          let dot = 0.0;
          for (let j = 0; j < 784; j++) {
            dot += state.model.weights[c][j] * x[j];
          }
          z[c] = dot + state.model.biases[c];
        }

        // 2. Softmax probabilities
        let maxZ = z[0];
        for (let c = 1; c < 10; c++) if (z[c] > maxZ) maxZ = z[c];

        let sumExp = 0.0;
        const probs = new Float32Array(10);
        for (let c = 0; c < 10; c++) {
          probs[c] = Math.exp(z[c] - maxZ);
          sumExp += probs[c];
        }
        for (let c = 0; c < 10; c++) probs[c] /= sumExp;

        // 3. Gradient dz = (probs - one_hot)
        for (let c = 0; c < 10; c++) {
          const oneHot = (c === y) ? 1.0 : 0.0;
          const dz = probs[c] - oneHot;

          // Update bias
          state.model.biases[c] -= lr * dz;

          // Update weights
          for (let j = 0; j < 784; j++) {
            state.model.weights[c][j] -= lr * dz * x[j];
          }
        }
      }
    }

    // Re-evaluate current input to see immediate accuracy improvement
    reprocessCurrent();

    if (navigator.vibrate) navigator.vibrate([40, 60, 40]);
    showToast(`Model fine-tuned on ${samples.length} samples (${epochs} epochs)!`);
  }

  // --- Modal (Deploy Guide) ---
  function initModal() {
    elements.deployGuideBtn.addEventListener('click', () => {
      elements.deployModal.showModal();
    });

    elements.closeDeployModalBtn.addEventListener('click', () => {
      elements.deployModal.close();
    });

    elements.gotItModalBtn.addEventListener('click', () => {
      elements.deployModal.close();
    });

    elements.deployModal.addEventListener('click', (e) => {
      const rect = elements.deployModal.getBoundingClientRect();
      if (
        e.clientX < rect.left ||
        e.clientX > rect.right ||
        e.clientY < rect.top ||
        e.clientY > rect.bottom
      ) {
        elements.deployModal.close();
      }
    });
  }

  // =========================================================================
  // THE COMPLETE MACHINE LEARNING IMAGE PIPELINE
  // =========================================================================

  function processSource(source, name) {
    if (!state.model) {
      console.warn('Model not yet loaded.');
      return;
    }

    state.currentSource = source;
    state.currentSourceName = name || 'Input Image';

    // -------------------------------------------------------------
    // STAGE 1: Raw Input Rendering & Metadata
    // -------------------------------------------------------------
    const rawCanvas = elements.rawPreviewCanvas;
    const rawCtx = rawCanvas.getContext('2d');
    const width = source.width || source.videoWidth || 280;
    const height = source.height || source.videoHeight || 280;

    rawCtx.clearRect(0, 0, rawCanvas.width, rawCanvas.height);
    rawCtx.fillStyle = '#000000';
    rawCtx.fillRect(0, 0, rawCanvas.width, rawCanvas.height);

    const aspect = width / height;
    let drawW = rawCanvas.width;
    let drawH = rawCanvas.height;
    let offsetX = 0;
    let offsetY = 0;

    if (aspect > 1) {
      drawH = rawCanvas.width / aspect;
      offsetY = (rawCanvas.height - drawH) / 2;
    } else {
      drawW = rawCanvas.height * aspect;
      offsetX = (rawCanvas.width - drawW) / 2;
    }

    rawCtx.drawImage(source, offsetX, offsetY, drawW, drawH);

    elements.rawSizeVal.textContent = `${width} × ${height}`;
    elements.rawSourceVal.textContent = name.length > 15 ? name.substring(0, 14) + '...' : name;

    // -------------------------------------------------------------
    // STAGE 2: Grayscale & Contrast Inversion
    // -------------------------------------------------------------
    const procCanvas = document.createElement('canvas');
    const maxSide = 280;
    const procScale = Math.min(1, maxSide / Math.max(width, height));
    procCanvas.width = Math.max(28, Math.round(width * procScale));
    procCanvas.height = Math.max(28, Math.round(height * procScale));

    const procCtx = procCanvas.getContext('2d', { willReadFrequently: true });
    procCtx.drawImage(source, 0, 0, procCanvas.width, procCanvas.height);

    const imgData = procCtx.getImageData(0, 0, procCanvas.width, procCanvas.height);
    const data = imgData.data;
    const totalPixels = procCanvas.width * procCanvas.height;

    // Detect perimeter background brightness to determine if white paper
    let perimeterSum = 0;
    let perimeterCount = 0;
    const pw = procCanvas.width;
    const ph = procCanvas.height;

    for (let x = 0; x < pw; x++) {
      const idxTop = (0 * pw + x) * 4;
      const idxBot = ((ph - 1) * pw + x) * 4;
      perimeterSum += (0.299 * data[idxTop] + 0.587 * data[idxTop + 1] + 0.114 * data[idxTop + 2]);
      perimeterSum += (0.299 * data[idxBot] + 0.587 * data[idxBot + 1] + 0.114 * data[idxBot + 2]);
      perimeterCount += 2;
    }

    for (let y = 1; y < ph - 1; y++) {
      const idxLeft = (y * pw + 0) * 4;
      const idxRight = (y * pw + (pw - 1)) * 4;
      perimeterSum += (0.299 * data[idxLeft] + 0.587 * data[idxLeft + 1] + 0.114 * data[idxLeft + 2]);
      perimeterSum += (0.299 * data[idxRight] + 0.587 * data[idxRight + 1] + 0.114 * data[idxRight + 2]);
      perimeterCount += 2;
    }

    const avgPerimeterBrightness = perimeterSum / Math.max(1, perimeterCount);

    let shouldInvert = false;
    if (state.settings.invertMode === 'auto') {
      shouldInvert = avgPerimeterBrightness > 120;
      elements.invertStateVal.textContent = shouldInvert ? 'Auto (Inverted)' : 'Auto (Kept)';
    } else if (state.settings.invertMode === 'invert') {
      shouldInvert = true;
      elements.invertStateVal.textContent = 'Forced Invert';
    } else {
      shouldInvert = false;
      elements.invertStateVal.textContent = 'Forced Keep';
    }

    const grayData = new Float32Array(totalPixels);
    const contrast = state.settings.contrast;

    for (let i = 0; i < totalPixels; i++) {
      const idx = i * 4;
      let gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];

      if (shouldInvert) {
        gray = 255.0 - gray;
      }

      gray = (gray - 128.0) * contrast + 128.0;
      gray = Math.max(0, Math.min(255, gray));

      grayData[i] = gray;

      data[idx] = gray;
      data[idx + 1] = gray;
      data[idx + 2] = gray;
      data[idx + 3] = 255;
    }

    procCtx.putImageData(imgData, 0, 0);

    // Render Stage 2 preview
    const grayCanvas = elements.grayCanvas;
    const grayCtx = grayCanvas.getContext('2d');
    grayCtx.clearRect(0, 0, grayCanvas.width, grayCanvas.height);
    grayCtx.fillStyle = '#000000';
    grayCtx.fillRect(0, 0, grayCanvas.width, grayCanvas.height);
    grayCtx.drawImage(procCanvas, 0, 0, grayCanvas.width, grayCanvas.height);

    // -------------------------------------------------------------
    // STAGE 3: 28x28 MNIST Standard Canvas Alignment
    // -------------------------------------------------------------
    const mnistCanvas = elements.mnistCanvas;
    const mnistCtx = mnistCanvas.getContext('2d', { willReadFrequently: true });
    mnistCtx.fillStyle = '#000000';
    mnistCtx.fillRect(0, 0, 28, 28);

    if (state.settings.alignMode === 'direct') {
      mnistCtx.drawImage(procCanvas, 0, 0, 28, 28);
      elements.stageAlignVal.textContent = 'Direct 28×28';
    } else {
      let minX = pw, maxX = -1, minY = ph, maxY = -1;
      const inkThreshold = 35;

      for (let y = 0; y < ph; y++) {
        for (let x = 0; x < pw; x++) {
          const val = grayData[y * pw + x];
          if (val > inkThreshold) {
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
        }
      }

      const boxW = maxX - minX + 1;
      const boxH = maxY - minY + 1;

      if (boxW > 3 && boxH > 3) {
        const digitCanvas = document.createElement('canvas');
        digitCanvas.width = boxW;
        digitCanvas.height = boxH;
        const digitCtx = digitCanvas.getContext('2d');
        digitCtx.drawImage(procCanvas, minX, minY, boxW, boxH, 0, 0, boxW, boxH);

        const scale = 20.0 / Math.max(boxW, boxH);
        const fitW = Math.max(1, Math.round(boxW * scale));
        const fitH = Math.max(1, Math.round(boxH * scale));

        const posX = Math.round((28 - fitW) / 2);
        const posY = Math.round((28 - fitH) / 2);

        mnistCtx.drawImage(digitCanvas, 0, 0, boxW, boxH, posX, posY, fitW, fitH);
        elements.stageAlignVal.textContent = 'Smart Center';
      } else {
        mnistCtx.drawImage(procCanvas, 0, 0, 28, 28);
        elements.stageAlignVal.textContent = 'Direct (Low Ink)';
      }
    }

    // -------------------------------------------------------------
    // STAGE 4: Extract 784 Normalized Input Vector
    // -------------------------------------------------------------
    const finalImgData = mnistCtx.getImageData(0, 0, 28, 28);
    const finalPixels = finalImgData.data;
    const inputVector = new Float32Array(784);
    let nonZeroCount = 0;

    for (let i = 0; i < 784; i++) {
      const r = finalPixels[i * 4];
      const g = finalPixels[i * 4 + 1];
      const b = finalPixels[i * 4 + 2];
      const val = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;

      inputVector[i] = val;
      if (val > 0.05) nonZeroCount++;
    }

    state.currentInputVector = inputVector;
    elements.nonZeroCount.textContent = `${nonZeroCount} / 784`;

    renderVectorHeatmap(inputVector);

    // -------------------------------------------------------------
    // STAGE 5 & 6: Neural Forward Pass (W · x + b) & Softmax
    // -------------------------------------------------------------
    const weights = state.model.weights;
    const biases = state.model.biases;

    const logits = new Float32Array(10);
    const dotProducts = new Float32Array(10);

    for (let c = 0; c < 10; c++) {
      const rowWeights = weights[c];
      let dot = 0.0;
      for (let j = 0; j < 784; j++) {
        dot += rowWeights[j] * inputVector[j];
      }
      dotProducts[c] = dot;
      logits[c] = dot + biases[c];
    }

    let maxLogit = logits[0];
    for (let c = 1; c < 10; c++) {
      if (logits[c] > maxLogit) maxLogit = logits[c];
    }

    const expLogits = new Float32Array(10);
    let sumExp = 0.0;
    for (let c = 0; c < 10; c++) {
      expLogits[c] = Math.exp(logits[c] - maxLogit);
      sumExp += expLogits[c];
    }

    const probabilities = new Float32Array(10);
    let predictedClass = 0;
    let maxProb = 0.0;

    for (let c = 0; c < 10; c++) {
      probabilities[c] = expLogits[c] / sumExp;
      if (probabilities[c] > maxProb) {
        maxProb = probabilities[c];
        predictedClass = c;
      }
    }

    state.currentWinner = predictedClass;
    state.currentConfidence = maxProb;

    renderPredictionResults(predictedClass, probabilities, logits, dotProducts, biases);
  }

  // Render Vector Heatmap
  function renderVectorHeatmap(vector) {
    const canvas = elements.vectorHeatmapCanvas;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const cellW = canvas.width / 28;
    const cellH = canvas.height / 28;

    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const val = vector[y * 28 + x];
        if (val < 0.02) {
          ctx.fillStyle = '#0a0c10';
        } else {
          const intensity = Math.min(255, Math.round(val * 255));
          const r = Math.round(intensity * 0.3);
          const g = Math.round(intensity);
          const b = Math.round(intensity * 0.65);
          ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        }
        ctx.fillRect(x * cellW, y * cellH, cellW, cellH);
      }
    }
  }

  // Render Prediction Dashboard & Update Labeling Prompt
  function renderPredictionResults(winner, probs, logits, dotProducts, biases) {
    const confidencePct = (probs[winner] * 100).toFixed(1);

    // Hero Card
    elements.predictedNumber.textContent = winner;
    elements.predictedNumber.classList.remove('animate-pop');
    void elements.predictedNumber.offsetWidth;
    elements.predictedNumber.classList.add('animate-pop');

    elements.confidenceValue.textContent = `${confidencePct}%`;
    elements.distMaxNote.textContent = `Class ${winner} highest (${confidencePct}%)`;

    // Stage 6 Mini Output
    elements.miniPredictedNum.textContent = winner;
    elements.miniPredictedConf.textContent = `${confidencePct}%`;
    elements.stageWinnerVal.textContent = `Digit ${winner}`;

    // Update Quick Confirm Button in Data Labeling Card
    if (elements.quickDigitText) {
      elements.quickDigitText.textContent = winner;
    }

    // Reset any active selection on chips
    if (elements.digitChipsGrid) {
      elements.digitChipsGrid.querySelectorAll('.digit-chip').forEach(c => c.classList.remove('active-picked'));
    }

    // Top Candidates
    const ranked = Array.from({ length: 10 }, (_, i) => ({
      digit: i,
      prob: probs[i],
      logit: logits[i]
    })).sort((a, b) => b.prob - a.prob);

    elements.topThreeChips.innerHTML = ranked.slice(0, 3).map((item, idx) => `
      <span class="chip ${idx === 0 ? 'chip-rank1' : ''}">
        #${idx + 1}: <strong>${item.digit}</strong> (${(item.prob * 100).toFixed(1)}%)
      </span>
    `).join('');

    // Softmax Distribution Bars
    elements.probabilityBars.innerHTML = '';
    for (let digit = 0; digit < 10; digit++) {
      const p = probs[digit];
      const pct = (p * 100).toFixed(1);
      const isWinner = digit === winner;

      const row = document.createElement('div');
      row.className = `prob-row ${isWinner ? 'winner' : ''}`;
      row.innerHTML = `
        <span class="prob-digit">${digit}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${Math.max(1, pct)}%"></div>
        </div>
        <span class="prob-value">${pct}%</span>
      `;
      elements.probabilityBars.appendChild(row);
    }

    // Raw Logits Inspector Table
    elements.logitsTableBody.innerHTML = '';
    for (let c = 0; c < 10; c++) {
      const isWinner = c === winner;
      const tr = document.createElement('tr');
      if (isWinner) tr.className = 'winner-row';

      tr.innerHTML = `
        <td><strong>Digit ${c}</strong> ${isWinner ? '★' : ''}</td>
        <td>${biases[c].toFixed(4)}</td>
        <td>${dotProducts[c].toFixed(4)}</td>
        <td>${logits[c].toFixed(4)}</td>
        <td><strong>${(probs[c] * 100).toFixed(2)}%</strong></td>
      `;
      elements.logitsTableBody.appendChild(tr);
    }
  }

  // --- Boot Application on DOM Ready ---
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
  } else {
    initApp();
  }

})();
