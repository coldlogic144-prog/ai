/**
 * ColdLogic AI - Handwritten Digit Recognizer
 * 100% Client-Side Machine Learning Pipeline
 * Compatible with GitHub Pages (Zero Backend)
 */

(function () {
  'use strict';

  // --- Global Application State ---
  const state = {
    model: null,
    currentSource: null, // HTMLImageElement or HTMLCanvasElement
    currentSourceName: 'Sample 0',
    activeTab: 'upload',
    settings: {
      alignMode: 'smart', // 'smart' or 'direct'
      invertMode: 'auto', // 'auto', 'invert', 'keep'
      contrast: 1.4,
    },
    isDrawing: false,
    lastX: 0,
    lastY: 0,
    brushSize: 22,
    debounceTimer: null
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

    // Tabs
    tabUploadBtn: document.getElementById('tabUploadBtn'),
    tabDrawBtn: document.getElementById('tabDrawBtn'),
    tabSamplesBtn: document.getElementById('tabSamplesBtn'),
    paneUpload: document.getElementById('paneUpload'),
    paneDraw: document.getElementById('paneDraw'),
    paneSamples: document.getElementById('paneSamples'),

    // Upload Pane
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('fileInput'),
    browseFileBtn: document.getElementById('browseFileBtn'),

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
    heroPredictionCard: document.getElementById('heroPredictionCard'),
    predictedNumber: document.getElementById('predictedNumber'),
    confidenceValue: document.getElementById('confidenceValue'),
    topThreeChips: document.getElementById('topThreeChips'),
    inferenceBadge: document.getElementById('inferenceBadge'),
    distMaxNote: document.getElementById('distMaxNote'),
    probabilityBars: document.getElementById('probabilityBars'),
    logitsTableBody: document.getElementById('logitsTableBody'),

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

  // --- Initialize App ---
  async function initApp() {
    initModel();
    initTabs();
    initUpload();
    initDrawingCanvas();
    initSamples();
    initSettings();
    initModal();
    initPasteHandler();

    // Load initial sample digit (0)
    setTimeout(() => {
      loadInitialDigit();
    }, 100);
  }

  // --- Model Loader ---
  async function initModel() {
    try {
      if (window.MODEL_DATA && window.MODEL_DATA.weights && window.MODEL_DATA.biases) {
        state.model = window.MODEL_DATA;
        onModelLoaded();
        return;
      }

      // Fallback: fetch model.json
      elements.modelStatusText.textContent = 'Fetching model.json...';
      const response = await fetch('model.json');
      if (!response.ok) throw new Error('Failed to load model.json');
      state.model = await response.json();
      onModelLoaded();
    } catch (err) {
      console.error('Error loading model weights:', err);
      elements.modelStatusPill.classList.remove('status-ready');
      elements.modelStatusPill.style.borderColor = 'rgba(239, 68, 68, 0.4)';
      elements.modelStatusPill.style.color = '#ef4444';
      elements.modelStatusText.textContent = 'Model Loading Failed';
    }
  }

  function onModelLoaded() {
    elements.modelStatusPill.classList.add('status-ready');
    elements.modelStatusText.textContent = `Model Ready · 10×784 Weights · 10 Biases`;
  }

  // --- Tab Navigation ---
  function initTabs() {
    const tabs = [
      { btn: elements.tabUploadBtn, pane: elements.paneUpload, id: 'upload' },
      { btn: elements.tabDrawBtn, pane: elements.paneDraw, id: 'draw' },
      { btn: elements.tabSamplesBtn, pane: elements.paneSamples, id: 'samples' }
    ];

    tabs.forEach(tab => {
      tab.btn.addEventListener('click', () => {
        tabs.forEach(t => {
          t.btn.classList.remove('active');
          t.btn.setAttribute('aria-selected', 'false');
          t.pane.classList.remove('active');
        });

        tab.btn.classList.add('active');
        tab.btn.setAttribute('aria-selected', 'true');
        tab.pane.classList.add('active');
        state.activeTab = tab.id;

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
            handleFile(blob, 'Pasted Clipboard Image');
            // Switch to upload tab
            elements.tabUploadBtn.click();
            break;
          }
        }
      }
    });
  }

  function handleFile(file, customName) {
    if (!file.type.startsWith('image/')) {
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

  // --- Interactive Drawing Canvas ---
  function initDrawingCanvas() {
    const canvas = elements.drawingCanvas;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    // Fill with black background
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
      const clientX = e.clientX ?? (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
      const clientY = e.clientY ?? (e.touches && e.touches[0] ? e.touches[0].clientY : 0);

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
      e.preventDefault();

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

    // Pointer / mouse / touch events
    canvas.addEventListener('pointerdown', startDrawing);
    canvas.addEventListener('pointermove', draw);
    window.addEventListener('pointerup', stopDrawing);
    window.addEventListener('pointercancel', stopDrawing);
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

    // Close on backdrop click
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

    // Aspect ratio fit
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
    // Process on an internal intermediate canvas (e.g. max 280x280)
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
      // top & bottom rows
      const idxTop = (0 * pw + x) * 4;
      const idxBot = ((ph - 1) * pw + x) * 4;
      perimeterSum += (0.299 * data[idxTop] + 0.587 * data[idxTop + 1] + 0.114 * data[idxTop + 2]);
      perimeterSum += (0.299 * data[idxBot] + 0.587 * data[idxBot + 1] + 0.114 * data[idxBot + 2]);
      perimeterCount += 2;
    }

    for (let y = 1; y < ph - 1; y++) {
      // left & right columns
      const idxLeft = (y * pw + 0) * 4;
      const idxRight = (y * pw + (pw - 1)) * 4;
      perimeterSum += (0.299 * data[idxLeft] + 0.587 * data[idxLeft + 1] + 0.114 * data[idxLeft + 2]);
      perimeterSum += (0.299 * data[idxRight] + 0.587 * data[idxRight + 1] + 0.114 * data[idxRight + 2]);
      perimeterCount += 2;
    }

    const avgPerimeterBrightness = perimeterSum / Math.max(1, perimeterCount);

    // Decision: should we invert?
    let shouldInvert = false;
    if (state.settings.invertMode === 'auto') {
      shouldInvert = avgPerimeterBrightness > 120; // light background, dark ink -> invert
      elements.invertStateVal.textContent = shouldInvert ? 'Auto (Inverted)' : 'Auto (Kept)';
    } else if (state.settings.invertMode === 'invert') {
      shouldInvert = true;
      elements.invertStateVal.textContent = 'Forced Invert';
    } else {
      shouldInvert = false;
      elements.invertStateVal.textContent = 'Forced Keep';
    }

    // Convert pixels to grayscale and apply contrast
    const grayData = new Float32Array(totalPixels);
    const contrast = state.settings.contrast;

    for (let i = 0; i < totalPixels; i++) {
      const idx = i * 4;
      let gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];

      if (shouldInvert) {
        gray = 255.0 - gray;
      }

      // Apply contrast boost centered around 128
      gray = (gray - 128.0) * contrast + 128.0;
      gray = Math.max(0, Math.min(255, gray));

      grayData[i] = gray;

      // Update image data for display on Stage 2 canvas
      data[idx] = gray;
      data[idx + 1] = gray;
      data[idx + 2] = gray;
      data[idx + 3] = 255;
    }

    procCtx.putImageData(imgData, 0, 0);

    // Render to Stage 2 preview
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
      // Direct resize: simply scale the grayscale canvas to 28x28
      mnistCtx.drawImage(procCanvas, 0, 0, 28, 28);
      elements.stageAlignVal.textContent = 'Direct 28×28';
    } else {
      // Smart MNIST Centering:
      // 1. Threshold to find bounding box of digit ink
      let minX = pw, maxX = -1, minY = ph, maxY = -1;
      const inkThreshold = 35; // noise floor

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
        // Cut out digit
        const digitCanvas = document.createElement('canvas');
        digitCanvas.width = boxW;
        digitCanvas.height = boxH;
        const digitCtx = digitCanvas.getContext('2d');
        digitCtx.drawImage(procCanvas, minX, minY, boxW, boxH, 0, 0, boxW, boxH);

        // Fit digit into a 20x20 box inside the 28x28 canvas (MNIST specification)
        const scale = 20.0 / Math.max(boxW, boxH);
        const fitW = Math.max(1, Math.round(boxW * scale));
        const fitH = Math.max(1, Math.round(boxH * scale));

        // Center inside 28x28
        const posX = Math.round((28 - fitW) / 2);
        const posY = Math.round((28 - fitH) / 2);

        mnistCtx.drawImage(digitCanvas, 0, 0, boxW, boxH, posX, posY, fitW, fitH);
        elements.stageAlignVal.textContent = 'Smart Center';
      } else {
        // Fallback if no digit ink detected
        mnistCtx.drawImage(procCanvas, 0, 0, 28, 28);
        elements.stageAlignVal.textContent = 'Direct (Low Ink)';
      }
    }

    // -------------------------------------------------------------
    // STAGE 4: Extract 784 Normalized Input Vector
    // -------------------------------------------------------------
    const finalImgData = mnistCtx.getImageData(0, 0, 28, 28);
    const finalPixels = finalImgData.data;
    const inputVector = new Float32Array(784); // 28 * 28 = 784
    let nonZeroCount = 0;

    for (let i = 0; i < 784; i++) {
      const r = finalPixels[i * 4];
      const g = finalPixels[i * 4 + 1];
      const b = finalPixels[i * 4 + 2];
      const val = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0; // Normalized [0.0, 1.0]

      inputVector[i] = val;
      if (val > 0.05) nonZeroCount++;
    }

    elements.nonZeroCount.textContent = `${nonZeroCount} / 784`;

    // Render Vector Mini-Heatmap on Stage 4
    renderVectorHeatmap(inputVector);

    // -------------------------------------------------------------
    // STAGE 5 & 6: Neural Forward Pass (W · x + b) & Softmax
    // -------------------------------------------------------------
    const weights = state.model.weights; // shape: [10][784]
    const biases = state.model.biases;   // shape: [10]

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

    // Softmax probabilities: P(c) = exp(z_c - max(z)) / sum(exp(z_k - max(z)))
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

    // -------------------------------------------------------------
    // Render Results & UI Visualizations
    // -------------------------------------------------------------
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
        // Blue to Cyan/Emerald gradient for active features
        if (val < 0.01) {
          ctx.fillStyle = '#080c14';
        } else {
          const r = Math.round(99 * (1 - val));
          const g = Math.round(102 + (185 - 102) * val);
          const b = Math.round(241 * val);
          ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        }
        ctx.fillRect(x * cellW, y * cellH, cellW, cellH);
      }
    }
  }

  // Render Prediction Dashboard
  function renderPredictionResults(winner, probs, logits, dotProducts, biases) {
    const confidencePct = (probs[winner] * 100).toFixed(1);

    // Hero Card
    elements.predictedNumber.textContent = winner;
    elements.predictedNumber.classList.remove('animate-pop');
    void elements.predictedNumber.offsetWidth; // Trigger reflow
    elements.predictedNumber.classList.add('animate-pop');

    elements.confidenceValue.textContent = `${confidencePct}%`;
    elements.distMaxNote.textContent = `Class ${winner} highest (${confidencePct}%)`;

    // Stage 6 Mini Output
    elements.miniPredictedNum.textContent = winner;
    elements.miniPredictedConf.textContent = `${confidencePct}%`;
    elements.stageWinnerVal.textContent = `Digit ${winner}`;

    // Top 3 Candidates
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
