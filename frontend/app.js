import { analyzeXray, getMockResponse } from './api.js';

const dom = {
    input: document.getElementById('xrayInput'),
    uploadZone: document.getElementById('uploadZone'),
    previewWrap: document.getElementById('previewWrap'),
    previewImage: document.getElementById('previewImage'),
    questionInput: document.getElementById('questionInput'),
    chatQuestionInput: document.getElementById('chatQuestionInput'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    chatAnalyzeBtn: document.getElementById('chatAnalyzeBtn'),
    progressCard: document.getElementById('progressCard'),
    stepItems: Array.from(document.querySelectorAll('.step-item')),
    emptyState: document.getElementById('emptyState'),
    resultsPanel: document.getElementById('resultsPanel'),
    severityRow: document.getElementById('severityRow'),
    severityDot: document.getElementById('severityDot'),
    severityLabel: document.getElementById('severityLabel'),
    severityDesc: document.getElementById('severityDesc'),
    captionBox: document.getElementById('captionBox'),
    findingsCount: document.getElementById('findingsCount'),
    avgConfidence: document.getElementById('avgConfidence'),
    inferenceTime: document.getElementById('inferenceTime'),
    originalImage: document.getElementById('originalImage'),
    heatmapImage: document.getElementById('heatmapImage'),
    chatStream: document.getElementById('chatStream'),
    modalOverlay: document.getElementById('modalOverlay'),
    modalImage: document.getElementById('modalImage'),
    modalClose: document.getElementById('modalClose'),
    modalDismiss: document.getElementById('modalDismiss'),
    modalDownload: document.getElementById('modalDownload'),
    exportReportBtn: document.getElementById('exportReportBtn'),
    reportTemplate: document.getElementById('reportTemplate'),
    reportOriginalImage: document.getElementById('reportOriginalImage'),
    reportHeatmapImage: document.getElementById('reportHeatmapImage'),
    reportCaption: document.getElementById('reportCaption')
};

const state = {
    file: null,
    previewUrl: '',
    response: null,
    rawResponse: null,
    requestTimer: null,
    activeStep: 0,
    analyzing: false,
    currentHeatmapUrl: '',
    currentOriginalUrl: '',
    inferenceMsFallback: null,
    lastQuestion: '',
    chatMessages: []
};

function toast(title, text, type = 'info', timeout = 4000) {
    return { title, text, type, timeout };
}

function setAnalyzeEnabled(enabled) {
    dom.analyzeBtn.disabled = !enabled;
    if (dom.chatAnalyzeBtn) {
        dom.chatAnalyzeBtn.disabled = !enabled;
    }
    dom.analyzeBtn.querySelector('span').textContent = 'Phân tích';
    if (dom.chatAnalyzeBtn) {
        dom.chatAnalyzeBtn.querySelector('span').textContent = state.response ? 'Gửi lại & phân tích' : 'Gửi & phân tích';
    }
}

function updateExportButton() {
    if (dom.exportReportBtn) {
        dom.exportReportBtn.disabled = !state.response;
    }
}

function resetSteps() {
    dom.stepItems.forEach((item) => item.classList.remove('is-active', 'is-done'));
    state.activeStep = 0;
}

function setStep(stepIndex) {
    dom.stepItems.forEach((item, index) => {
        item.classList.toggle('is-active', index === stepIndex);
        item.classList.toggle('is-done', index < stepIndex);
    });
    state.activeStep = stepIndex;
}

function finishSteps() {
    dom.stepItems.forEach((item) => item.classList.add('is-done'));
}

function showProgress(show) {
    dom.progressCard.classList.toggle('hidden', !show);
}

function normalizeSeverity(severity) {
    const map = {
        normal: { label: 'BÌNH THƯỜNG', desc: 'Không ghi nhận bất thường nổi bật.', color: '#059669', shadow: 'rgba(5,150,105,0.25)' },
        low: { label: 'THẤP', desc: 'Mức cảnh báo thấp, cần đối chiếu thêm lâm sàng.', color: '#1A73E8', shadow: 'rgba(26,115,232,0.25)' },
        moderate: { label: 'TRUNG BÌNH', desc: 'Có dấu hiệu bất thường đáng chú ý.', color: '#D97706', shadow: 'rgba(217,119,6,0.25)' },
        high: { label: 'CAO', desc: 'Mức độ đáng lo ngại, cần đánh giá chuyên sâu.', color: '#DC2626', shadow: 'rgba(220,38,38,0.25)' }
    };
    return map[String(severity || 'normal').toLowerCase()] || map.normal;
}

function formatPercent(value) {
    return `${Math.round((Number(value) || 0) * 100)}%`;
}

function formatMs(value) {
    const ms = Number(value);
    return Number.isFinite(ms) ? `${Math.round(ms)} ms` : '—';
}

function escapeHtml(text) {
    return String(text)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function formatChatText(text) {
    const normalized = String(text || '')
        .replace(/\r\n/g, '\n')
        .replace(/\*\*(.*?)\*\*/g, '$1')
        .replace(/__(.*?)__/g, '$1')
        .replace(/\*(.*?)\*/g, '$1')
        .replace(/^[-*•]\s+/gm, '• ')
        .trim();

    return escapeHtml(normalized).replaceAll('\n', '<br />');
}

function getDetections(response) {
    if (Array.isArray(response?.detection)) {
        return response.detection;
    }
    if (Array.isArray(response?.detections)) {
        return response.detections;
    }
    if (Array.isArray(response?.pathologies)) {
        return response.pathologies;
    }
    return [];
}

function loadPreview(file) {
    if (state.previewUrl) {
        URL.revokeObjectURL(state.previewUrl);
    }
    state.previewUrl = URL.createObjectURL(file);
    dom.previewImage.src = state.previewUrl;
    dom.previewImage.alt = file.name;
    dom.previewWrap.classList.remove('hidden');
}

function renderOriginalImage(sourceUrl) {
    const finalUrl = sourceUrl || state.previewUrl;
    if (!finalUrl) return;
    dom.originalImage.src = finalUrl;
    dom.originalImage.alt = 'Ảnh X-quang gốc';
}

function renderHeatmapImage(response) {
    const heatmapUrl = response?.heatmap_url || '';
    const base64 = response?.heatmap?.gradcam_base64;

    if (heatmapUrl) {
        dom.heatmapImage.src = heatmapUrl;
        dom.heatmapImage.alt = 'Ảnh GradCAM và BBox';
        state.currentHeatmapUrl = heatmapUrl;
        return;
    }

    if (base64) {
        const prefix = base64.startsWith('data:') ? base64 : `data:image/png;base64,${base64}`;
        dom.heatmapImage.src = prefix;
        dom.heatmapImage.alt = 'Ảnh GradCAM và BBox';
        state.currentHeatmapUrl = prefix;
        return;
    }

    state.currentHeatmapUrl = '';
    const canvas = document.createElement('canvas');
    canvas.width = 1200;
    canvas.height = 900;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0d1b2a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#A8C0D6';
    ctx.font = '600 42px IBM Plex Mono';
    ctx.fillText('Bản đồ nhiệt không khả dụng', 60, 100);
    ctx.font = '400 26px Outfit';
    ctx.fillText('Backend chưa trả về ảnh overlay.', 60, 150);
    dom.heatmapImage.src = canvas.toDataURL('image/png');
    dom.heatmapImage.alt = 'Bản đồ nhiệt không khả dụng';
}

function renderSeverity(response) {
    const info = normalizeSeverity(response?.severity);
    dom.severityLabel.textContent = info.label;
    dom.severityDesc.textContent = info.desc;
    dom.severityDot.style.background = info.color;
    dom.severityDot.style.boxShadow = `0 0 0 0 ${info.shadow}`;
    dom.severityRow.style.borderColor = info.color;
}

function renderCaption(response) {
    dom.captionBox.textContent = response?.caption || 'Chưa có mô tả y khoa được sinh ra.';
}

function renderStats(response) {
    const detections = getDetections(response);

    const confValues = detections.map((item) => Number(item.confidence)).filter((value) => Number.isFinite(value));
    const avg = confValues.length ? confValues.reduce((sum, value) => sum + value, 0) / confValues.length : 0;
    const findings = detections.length;
    const ms = response?.inference_ms ?? state.inferenceMsFallback ?? 0;

    dom.findingsCount.textContent = String(findings);
    dom.avgConfidence.textContent = formatPercent(avg);
    dom.inferenceTime.textContent = formatMs(ms);
}


function renderResults(response) {
    state.response = response;
    dom.emptyState.classList.add('hidden');
    dom.resultsPanel.classList.remove('hidden');

    renderSeverity(response);
    renderCaption(response);
    renderStats(response);
    renderHeatmapImage(response);
}

function renderChatHistory() {
    if (!state.chatMessages.length) {
        dom.chatStream.innerHTML = '<div class="chat-empty">Nhập câu hỏi lâm sàng để bắt đầu cuộc hội thoại.</div>';
        return;
    }

    dom.chatStream.innerHTML = state.chatMessages.map((message) => {
        const escapedText = formatChatText(message.text);
        if (message.role === 'user') {
            return `<div class="chat-message chat-message--user"><div class="chat-bubble chat-bubble--user">${escapedText}</div></div>`;
        }
        return `<div class="chat-message chat-message--assistant"><div class="chat-bubble chat-bubble--assistant">${escapedText}</div></div>`;
    }).join('');

    dom.chatStream.scrollTop = dom.chatStream.scrollHeight;
}

function openModal(src, downloadUrl = '') {
    if (!src) return;
    dom.modalImage.src = src;
    dom.modalDownload.href = downloadUrl || src;
    dom.modalOverlay.classList.remove('hidden');
    dom.modalOverlay.setAttribute('aria-hidden', 'false');
}

function closeModal() {
    dom.modalOverlay.classList.add('hidden');
    dom.modalOverlay.setAttribute('aria-hidden', 'true');
    dom.modalImage.src = '';
}

function exportReport() {
    if (!state.response) {
        console.log('No response available');
        return;
    }

    console.log('Exporting report with caption:', state.response.caption);
    console.log('Original image:', dom.originalImage.src);
    console.log('Heatmap image:', dom.heatmapImage.src);

    // Convert images to base64
    async function convertAndExport() {
        try {
            // Convert original image (blob URL) to data URL
            const originalDataUrl = await imageUrlToBase64(dom.originalImage.src);
            console.log('Original image converted to base64');

            // Convert heatmap image to data URL
            const heatmapDataUrl = await imageUrlToBase64(dom.heatmapImage.src);
            console.log('Heatmap image converted to base64');

            // Send to backend for PDF generation
            const formData = new FormData();
            formData.append('originalImageBase64', originalDataUrl);
            formData.append('heatmapImageBase64', heatmapDataUrl);
            formData.append('caption', state.response.caption || 'Chưa có mô tả y khoa được sinh ra.');

            console.log('Sending to backend...');

            const response = await fetch('/export-pdf', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Backend error: ${response.status}`);
            }

            // Get PDF blob and download
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ChestVision_Report.pdf';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            console.log('PDF downloaded successfully');

        } catch (err) {
            console.error('Error in exportReport:', err);
            alert('Lỗi xuất báo cáo: ' + err.message);
        }
    }

    function imageUrlToBase64(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                resolve(canvas.toDataURL('image/jpeg', 0.95));
            };
            img.onerror = () => reject(new Error('Failed to load image: ' + url));
            img.src = url;
        });
    }

    convertAndExport();
}

function startFakeProgress() {
    resetSteps();
    showProgress(true);
    let step = 0;
    setStep(step);
    clearProgressTimer();
    state.requestTimer = window.setInterval(() => {
        step = Math.min(step + 1, dom.stepItems.length - 1);
        setStep(step);
        if (step >= dom.stepItems.length - 1) {
            clearProgressTimer();
        }
    }, 900);
}

function clearProgressTimer() {
    if (state.requestTimer) {
        clearInterval(state.requestTimer);
        state.requestTimer = null;
    }
}

function setBusy(isBusy) {
    state.analyzing = isBusy;
    dom.previewWrap.classList.toggle('analyzing', isBusy);
    dom.analyzeBtn.disabled = isBusy || !state.file;
    if (dom.chatAnalyzeBtn) {
        dom.chatAnalyzeBtn.disabled = isBusy || !state.file;
    }
    dom.analyzeBtn.innerHTML = isBusy
        ? '<i class="fa-solid fa-bolt"></i><span>Đang phân tích...</span>'
        : `<i class="fa-solid fa-microscope"></i><span>${state.response ? 'Phân tích' : 'Phân tích'}</span>`;
    if (dom.chatAnalyzeBtn) {
        dom.chatAnalyzeBtn.innerHTML = isBusy
            ? '<i class="fa-solid fa-bolt"></i><span>Đang phân tích...</span>'
            : `<i class="fa-solid fa-comments"></i><span>${state.response ? 'Gửi lại & phân tích' : 'Gửi & phân tích'}</span>`;
    }
}

function normalizeResponse(response) {
    const detections = getDetections(response).map((item) => ({
        class: item.class || item.name,
        confidence: item.confidence,
        threshold: item.threshold,
        class_index: item.class_index
    }));

    const hasLesion = typeof response?.has_lesion === 'boolean'
        ? response.has_lesion
        : detections.length > 0;

    return {
        has_lesion: hasLesion,
        severity: response?.severity || 'normal',
        caption: response?.caption || '',
        detections,
        heatmap_url: response?.heatmap_url || '',
        answer: response?.answer || '',
        raw: response
    };
}

function renderChat(response) {
    const question = state.lastQuestion || dom.chatQuestionInput?.value.trim() || dom.questionInput?.value.trim() || '';
    const answer = response?.answer || '';

    if (question && !state.chatMessages.some((message) => message.role === 'user' && message.text === question)) {
        state.chatMessages.push({ role: 'user', text: question });
    }

    if (answer) {
        state.chatMessages.push({ role: 'assistant', text: answer });
    } else if (question && state.chatMessages.length && state.chatMessages[state.chatMessages.length - 1].role === 'user') {
        state.chatMessages.push({ role: 'assistant', text: 'Chưa có câu trả lời từ backend.' });
    }

    renderChatHistory();
}

async function handleAnalyze() {
    if (!state.file) {
        toast('Thiếu ảnh', 'Vui lòng chọn một file ảnh X-quang trước khi phân tích.', 'error');
        return;
    }

    // Đọc từ input phù hợp: nếu có từ chat input (follow-up) thì dung, không thì dùng left panel
    let question = dom.chatQuestionInput?.value.trim() || '';
    if (!question) {
        question = dom.questionInput?.value.trim() || '';
    }

    state.lastQuestion = question;
    if (question) {
        state.chatMessages.push({ role: 'user', text: question });
        renderChatHistory();
        // Xóa nội dung từ input được sử dụng
        if (dom.chatQuestionInput?.value.trim()) {
            dom.chatQuestionInput.value = '';
        } else if (dom.questionInput?.value.trim()) {
            dom.questionInput.value = '';
        }
    }
    setBusy(true);
    startFakeProgress();

    const startedAt = performance.now();

    try {
        const backendResponse = await analyzeXray(state.file, question);
        const response = normalizeResponse(backendResponse);
        state.rawResponse = backendResponse;
        state.inferenceMsFallback = Math.round(performance.now() - startedAt);

        state.currentOriginalUrl = state.previewUrl;
        renderOriginalImage(state.currentOriginalUrl);
        renderResults(response);
        updateExportButton();
        if (question || response.answer) {
            if (!question) {
                state.chatMessages.push({ role: 'assistant', text: response.answer || 'Không có câu trả lời.' });
            } else {
                state.chatMessages.push({ role: 'assistant', text: response.answer || 'Chưa có câu trả lời từ backend.' });
            }
            renderChatHistory();
        }
        finishSteps();
        window.setTimeout(() => {
            dom.resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 150);
    } catch (error) {
        const fallback = state.file ? getMockResponse() : null;
        if (fallback) {
            fallback.caption = 'Backend chưa phản hồi. Đang hiển thị mock response để kiểm thử UI.';
            fallback.heatmap.original_url = state.previewUrl;
            state.rawResponse = fallback;
            state.inferenceMsFallback = fallback.inference_ms;
            renderOriginalImage(state.previewUrl);
            renderResults(fallback);
            updateExportButton();
            if (question) {
                state.chatMessages.push({ role: 'assistant', text: fallback.answer || 'Backend chưa phản hồi.' });
                renderChatHistory();
            }
            finishSteps();
            toast('Backend chưa sẵn sàng', 'Đã dùng mock response để tiếp tục kiểm thử giao diện.', 'info');
        } else {
            toast('Lỗi phân tích', error.message || 'Không thể kết nối backend.', 'error');
            dom.resultsPanel.classList.add('hidden');
            dom.emptyState.classList.remove('hidden');
        }
    } finally {
        clearProgressTimer();
        setBusy(false);
    }
}

function handleFile(file) {
    if (!file) return;
    state.file = file;
    state.response = null;
    loadPreview(file);
    renderOriginalImage(state.previewUrl);
    setAnalyzeEnabled(true);
    updateExportButton();
    toast('Ảnh đã tải', file.name, 'success', 2500);
}

dom.input.addEventListener('change', (event) => {
    handleFile(event.target.files?.[0]);
});

dom.uploadZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dom.uploadZone.classList.add('is-dragover');
});

dom.uploadZone.addEventListener('dragleave', () => {
    dom.uploadZone.classList.remove('is-dragover');
});

dom.uploadZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dom.uploadZone.classList.remove('is-dragover');
    const file = event.dataTransfer.files?.[0];
    if (file) {
        const transfer = new DataTransfer();
        transfer.items.add(file);
        dom.input.files = transfer.files;
        handleFile(file);
    }
});

dom.analyzeBtn.addEventListener('click', handleAnalyze);

if (dom.chatAnalyzeBtn) {
    dom.chatAnalyzeBtn.addEventListener('click', handleAnalyze);
}

if (dom.exportReportBtn) {
    dom.exportReportBtn.addEventListener('click', exportReport);
}

if (dom.questionInput) {
    dom.questionInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleAnalyze();
        }
    });
}

if (dom.chatQuestionInput) {
    dom.chatQuestionInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleAnalyze();
        }
    });
}

dom.modalOverlay.addEventListener('click', (event) => {
    if (event.target === dom.modalOverlay) closeModal();
});
dom.modalClose.addEventListener('click', closeModal);
dom.modalDismiss.addEventListener('click', closeModal);

dom.originalImage.addEventListener('click', () => {
    if (dom.originalImage.src) openModal(dom.originalImage.src, dom.originalImage.src);
});

dom.heatmapImage.addEventListener('click', () => {
    if (dom.heatmapImage.src) openModal(dom.heatmapImage.src, dom.heatmapImage.src);
});

window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeModal();
});

setAnalyzeEnabled(false);
renderChatHistory();
