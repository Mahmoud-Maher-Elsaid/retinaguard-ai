const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const predictBtn = document.getElementById("predictBtn");

const resultTitle = document.getElementById("resultTitle");
const statusPill = document.getElementById("statusPill");
const predictedGrade = document.getElementById("predictedGrade");
const confidence = document.getElementById("confidence");
const riskScore = document.getElementById("riskScore");
const triageDecision = document.getElementById("triageDecision");
const probabilityBars = document.getElementById("probabilityBars");

const detailsSection = document.querySelector(".details-section");

detailsSection.innerHTML = `
    <h4>Clinical Decision Report</h4>

    <div class="clinical-report">
        <div class="report-card">
            <span>Uncertainty Level</span>
            <strong id="reportUncertainty">--</strong>
        </div>

        <div class="report-card">
            <span>Image Quality</span>
            <strong id="reportImageQuality">--</strong>
        </div>

        <div class="report-card">
            <span>Lesion Evidence</span>
            <strong id="reportLesionEvidence">--</strong>
        </div>

        <div class="report-card">
            <span>Grade Consistency</span>
            <strong id="reportConsistency">--</strong>
        </div>

        <div class="report-card">
            <span>Total Lesion Burden</span>
            <strong id="reportBurden">--</strong>
        </div>

        <div class="report-card">
            <span>Device</span>
            <strong id="reportDevice">--</strong>
        </div>
    </div>

    <div class="lesion-panel">
        <h4>Predicted Lesion Statistics</h4>
        <div id="lesionCards" class="lesion-cards"></div>
    </div>

    <div class="narrative-box">
        <h4>AI Clinical Summary</h4>
        <p id="clinicalNarrative">
            Upload an image to generate a structured clinical-style summary.
        </p>
    </div>
`;

const reportUncertainty = document.getElementById("reportUncertainty");
const reportImageQuality = document.getElementById("reportImageQuality");
const reportLesionEvidence = document.getElementById("reportLesionEvidence");
const reportConsistency = document.getElementById("reportConsistency");
const reportBurden = document.getElementById("reportBurden");
const reportDevice = document.getElementById("reportDevice");
const lesionCards = document.getElementById("lesionCards");
const clinicalNarrative = document.getElementById("clinicalNarrative");

let selectedFile = null;

const gradeNames = {
    "0": "No DR",
    "1": "Mild",
    "2": "Moderate",
    "3": "Severe",
    "4": "Proliferative DR"
};

function setStatus(text, type) {
    statusPill.textContent = text;
    statusPill.className = `status-pill ${type}`;
}

function formatPercent(value) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "--";
    }

    return `${(value * 100).toFixed(2)}%`;
}

function formatNumber(value, digits = 3) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "--";
    }

    return Number(value).toFixed(digits);
}

function prettyText(value) {
    if (!value) {
        return "--";
    }

    return String(value)
        .replaceAll("_", " ")
        .replace(/\b\w/g, char => char.toUpperCase());
}

function decisionClass(decision) {
    if (!decision) {
        return "decision-neutral";
    }

    if (decision.includes("manual")) {
        return "decision-review";
    }

    if (decision.includes("urgent")) {
        return "decision-urgent";
    }

    if (decision.includes("safe")) {
        return "decision-safe";
    }

    return "decision-followup";
}

function renderProbabilityBars(probabilities) {
    probabilityBars.innerHTML = "";

    Object.entries(probabilities).forEach(([label, info]) => {
        const probability = info.probability;

        const row = document.createElement("div");
        row.className = "prob-row";

        row.innerHTML = `
            <div class="prob-label">
                <span>${label} - ${info.label_name || gradeNames[label]}</span>
                <span>${formatPercent(probability)}</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width: ${probability * 100}%"></div>
            </div>
        `;

        probabilityBars.appendChild(row);
    });
}

function renderLesionCards(lesions) {
    lesionCards.innerHTML = "";

    const lesionNames = {
        microaneurysms: "Microaneurysms",
        haemorrhages: "Haemorrhages",
        hard_exudates: "Hard Exudates",
        soft_exudates: "Soft Exudates"
    };

    Object.entries(lesions).forEach(([lesionKey, info]) => {
        const card = document.createElement("div");
        card.className = "lesion-card";

        card.innerHTML = `
            <span>${lesionNames[lesionKey] || prettyText(lesionKey)}</span>
            <strong>${formatPercent(info.area_ratio)}</strong>
            <small>Max probability: ${formatPercent(info.probability_max)}</small>
        `;

        lesionCards.appendChild(card);
    });
}

function buildNarrative(data) {
    const cls = data.classification;
    const safety = data.safety_gate;
    const quality = data.image_quality;
    const segmentation = data.segmentation;

    const grade = `${cls.predicted_label} - ${cls.predicted_label_name}`;
    const triage = prettyText(safety.triage_decision);
    const uncertainty = prettyText(safety.uncertainty_level);
    const imageQuality = prettyText(quality.image_quality_status);
    const lesionEvidence = prettyText(safety.lesion_evidence_level);
    const consistency = prettyText(safety.lesion_grade_consistency);

    return `
        RetinaGuard-AI predicted diabetic retinopathy grade ${grade} with confidence ${formatPercent(cls.confidence)}.
        The Safety Gate classified this case as "${triage}".
        Uncertainty level is ${uncertainty}, image quality is ${imageQuality}, lesion evidence is ${lesionEvidence},
        and lesion-grade consistency is ${consistency}.
        The total predicted lesion burden is ${formatPercent(segmentation.total_lesion_union_area_ratio)}.
        This result is research-only and should not be used as a clinical diagnosis.
    `.trim();
}

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];

    if (!file) {
        return;
    }

    selectedFile = file;

    const reader = new FileReader();

    reader.onload = (event) => {
        previewImage.src = event.target.result;
        previewImage.style.display = "block";
        previewPlaceholder.style.display = "none";
    };

    reader.readAsDataURL(file);

    resultTitle.textContent = "Ready to analyze";
    setStatus("Ready", "idle");
});

predictBtn.addEventListener("click", async () => {
    if (!selectedFile) {
        alert("Please choose a retinal image first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    setStatus("Running", "running");
    resultTitle.textContent = "Analyzing image...";
    predictBtn.disabled = true;
    predictBtn.textContent = "Running analysis...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Prediction failed");
        }

        const cls = data.classification;
        const safety = data.safety_gate;
        const quality = data.image_quality;
        const segmentation = data.segmentation;

        resultTitle.textContent = "Analysis completed";
        setStatus("Completed", "done");

        predictedGrade.textContent = `${cls.predicted_label} - ${cls.predicted_label_name}`;
        confidence.textContent = formatPercent(cls.confidence);
        riskScore.textContent = formatNumber(safety.risk_score, 3);

        triageDecision.textContent = prettyText(safety.triage_decision);
        triageDecision.className = decisionClass(safety.triage_decision);

        reportUncertainty.textContent = prettyText(safety.uncertainty_level);
        reportImageQuality.textContent = `${prettyText(quality.image_quality_status)} (${formatNumber(quality.quality_score, 3)})`;
        reportLesionEvidence.textContent = prettyText(safety.lesion_evidence_level);
        reportConsistency.textContent = prettyText(safety.lesion_grade_consistency);
        reportBurden.textContent = formatPercent(segmentation.total_lesion_union_area_ratio);
        reportDevice.textContent = data.device || "--";

        renderProbabilityBars(cls.probabilities);
        renderLesionCards(segmentation.lesions);

        clinicalNarrative.textContent = buildNarrative(data);

    } catch (error) {
        resultTitle.textContent = "Analysis failed";
        setStatus("Error", "error");

        clinicalNarrative.textContent = error.message;
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = "Run RetinaGuard Analysis";
    }
});
