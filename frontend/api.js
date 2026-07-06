const BASE_URL = '/api';

async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
        const errorText = await response.text().catch(() => '');
        throw new Error(`Yêu cầu thất bại (${response.status}): ${errorText || response.statusText}`);
    }
    return response.json();
}

async function tryEndpoints(endpoints, options = {}) {
    let lastError = null;
    for (const endpoint of endpoints) {
        try {
            return await requestJson(endpoint, options);
        } catch (error) {
            lastError = error;
        }
    }
    throw lastError || new Error('Yêu cầu API thất bại');
}

export async function analyzeXray(imageFile, question = '') {
    if (!imageFile) {
        throw new Error('Missing image file');
    }

    const formData = new FormData();
    formData.append('file', imageFile, imageFile.name);
    if (question && question.trim()) {
        formData.append('question', question.trim());
    }

    return tryEndpoints([
        '/analyze',
        `${BASE_URL}/analyze`
    ], {
        method: 'POST',
        body: formData
    });
}

export async function getHeatmap(jobId) {
    if (!jobId) {
        return null;
    }

    try {
        return await tryEndpoints([
            `/heatmap/${encodeURIComponent(jobId)}`,
            `${BASE_URL}/heatmap/${encodeURIComponent(jobId)}`
        ]);
    } catch {
        return null;
    }
}

export function getMockResponse() {
    return {
        job_id: 'mock-job',
        inference_ms: 1243,
        severity: 'low',
        caption: 'Trung thất rộng, tim to vừa, nghi ngờ phù mạch, không tràn dịch màng phổi rõ.',
        detections: [
            { class: 'Lung Opacity', confidence: 0.63, threshold: 0.60, bbox: [80, 70, 340, 330] },
            { class: 'Cardiomegaly', confidence: 0.62, threshold: 0.60, bbox: [120, 100, 360, 350] }
        ],
        heatmap: {
            original_url: null,
            gradcam_url: null,
            gradcam_base64: null
        },
        vqa_answer: 'Dựa trên ảnh và các dấu hiệu phát hiện được, chưa ghi nhận tràn dịch màng phổi rõ.'
    };
}

export { BASE_URL };