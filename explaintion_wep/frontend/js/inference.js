/**
 * BinX Week 8 Showcase — Interactive ML Lab
 * Real-time inference against the Flask backend
 */

document.addEventListener('DOMContentLoaded', () => {
    const API_BASE = 'http://localhost:5000/api';

    const labInput = document.getElementById('labInput');
    const labPredict = document.getElementById('labPredict');
    const labResult = document.getElementById('labResult');
    const labSteps = document.querySelectorAll('.lab-step');
    const labPredClass = document.getElementById('labPredClass');
    const labConfFill = document.getElementById('labConfFill');
    const labConfVal = document.getElementById('labConfVal');
    const labCleanedText = document.getElementById('labCleanedText');
    const labFeatures = document.getElementById('labFeatures');
    const labTiming = document.getElementById('labTiming');

    // Example buttons
    document.querySelectorAll('.lab-example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            labInput.value = btn.getAttribute('data-text');
            labInput.focus();
        });
    });

    // Predict button
    labPredict.addEventListener('click', runPrediction);

    // Enter key
    labInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            runPrediction();
        }
    });

    async function runPrediction() {
        const text = labInput.value.trim();
        if (!text) return;

        // Disable button, show loading
        labPredict.disabled = true;
        labPredict.querySelector('.btn-text').style.display = 'none';
        labPredict.querySelector('.btn-loading').style.display = 'inline-flex';

        // Show result panel
        labResult.style.display = 'block';

        // Reset steps
        labSteps.forEach(step => step.classList.remove('visible'));

        try {
            // Step 1: Input received
            await animateStep(0);

            // Step 2: Preprocessing
            await animateStep(1);

            // Step 3: Feature transformation
            await animateStep(2);

            // Step 4: Model inference
            const response = await fetch(`${API_BASE}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const result = await response.json();
            await animateStep(3);

            // Step 5: Explanation
            await animateStep(4);

            // Display results
            displayResults(result);

        } catch (error) {
            console.error('Prediction failed:', error);
            displayError(error.message);
        } finally {
            labPredict.disabled = false;
            labPredict.querySelector('.btn-text').style.display = 'inline';
            labPredict.querySelector('.btn-loading').style.display = 'none';
        }
    }

    async function animateStep(index) {
        return new Promise(resolve => {
            setTimeout(() => {
                if (labSteps[index]) {
                    labSteps[index].classList.add('visible');
                }
                resolve();
            }, 200 + index * 150);
        });
    }

    function displayResults(result) {
        // Prediction class
        const isPositive = result.prediction === 1;
        labPredClass.textContent = isPositive ? 'Positive (+)' : 'Negative (-)';
        labPredClass.className = 'lab-pred-class ' + (isPositive ? 'positive' : 'negative');

        // Confidence
        const conf = (result.confidence * 100).toFixed(1);
        labConfFill.style.width = conf + '%';
        labConfFill.style.background = isPositive
            ? 'linear-gradient(90deg, #10b981, #06b6d4)'
            : 'linear-gradient(90deg, #ef4444, #f59e0b)';
        labConfVal.textContent = conf + '%';

        // Cleaned text
        labCleanedText.textContent = result.cleaned || '(preprocessing unavailable)';

        // Features
        labFeatures.innerHTML = '';
        if (result.top_features && result.top_features.length > 0) {
            const maxWeight = Math.max(...result.top_features.map(f => Math.abs(f.weight)));

            result.top_features.slice(0, 8).forEach(f => {
                const item = document.createElement('div');
                item.className = `lab-feature-item ${f.direction}`;
                const barWidth = Math.abs(f.weight) / maxWeight * 100;
                item.innerHTML = `
                    <span class="lab-feature-word">${f.feature}</span>
                    <span class="lab-feature-weight">${f.weight > 0 ? '+' : ''}${f.weight.toFixed(4)}</span>
                    <div class="lab-feature-bar">
                        <div class="lab-feature-fill ${f.direction}" style="width: ${barWidth}%"></div>
                    </div>
                `;
                labFeatures.appendChild(item);
            });
        } else {
            labFeatures.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem;">No feature data available</p>';
        }

        // Timing
        if (result.timing) {
            labTiming.textContent = `Total: ${result.timing.total_ms}ms (preprocess: ${result.timing.preprocess_ms}ms, vectorize: ${result.timing.vectorize_ms}ms, predict: ${result.timing.predict_ms}ms)`;
        }
    }

    function displayError(message) {
        labPredClass.textContent = 'Error';
        labPredClass.className = 'lab-pred-class negative';
        labConfFill.style.width = '0%';
        labConfVal.textContent = '';
        labCleanedText.textContent = `Error: ${message}. Make sure the backend server is running on port 5000.`;
        labFeatures.innerHTML = '';
        labTiming.textContent = '';
    }

    // ============================================================
    // Check backend health on load
    // ============================================================
    async function checkBackend() {
        try {
            const resp = await fetch(`${API_BASE}/health`);
            const data = await resp.json();
            if (!data.model_loaded) {
                console.warn('Backend running but model not loaded');
            }
        } catch (e) {
            console.warn('Backend not reachable. Interactive lab will show errors.');
        }
    }
    checkBackend();
});
