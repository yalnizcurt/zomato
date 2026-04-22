/**
 * Restaurant Recommender — Frontend Application Logic
 *
 * Handles:
 *  - Loading metadata (cities, cuisines) from the API
 *  - Cuisine search/select with tags
 *  - Form submission → API call → results rendering
 */

// ================================================================
// DOM Elements
// ================================================================
const form         = document.getElementById("preference-form");
const submitBtn    = document.getElementById("submit-btn");
const locationSel  = document.getElementById("location");
const cuisineInput = document.getElementById("cuisines-input");
const cuisineDrop  = document.getElementById("cuisine-dropdown");
const selectedBox  = document.getElementById("selected-cuisines");
const ratingSlider = document.getElementById("min-rating");
const ratingValue  = document.getElementById("rating-value");
const formSection  = document.getElementById("form-section");
const resultsSection = document.getElementById("results-section");
const resultsGrid  = document.getElementById("results-grid");
const resultsTitle = document.getElementById("results-title");
const resultsSub   = document.getElementById("results-subtitle");
const backBtn      = document.getElementById("back-btn");
const errorToast   = document.getElementById("error-toast");
const errorMsg     = document.getElementById("error-message");
const errorClose   = document.getElementById("error-close");

// ================================================================
// State
// ================================================================
let allCuisines = [];
let selectedCuisines = [];

// ================================================================
// Initialisation — load metadata from API
// ================================================================
async function init() {
    try {
        const res = await fetch("/api/metadata");
        if (!res.ok) throw new Error("Failed to load metadata");

        const data = await res.json();

        // Populate city dropdown
        locationSel.innerHTML = '<option value="" disabled selected>Select your city</option>';
        data.cities.forEach(city => {
            const opt = document.createElement("option");
            opt.value = city;
            opt.textContent = capitalise(city);
            locationSel.appendChild(opt);
        });

        // Store cuisines for search
        allCuisines = data.cuisines;

    } catch (err) {
        showError("Could not load data. Please ensure the server is running and the dataset is available.");
        console.error(err);
    }
}

// ================================================================
// Cuisine search & tag system
// ================================================================
cuisineInput.addEventListener("input", () => {
    const query = cuisineInput.value.trim().toLowerCase();
    if (!query) {
        cuisineDrop.classList.remove("show");
        return;
    }

    const matches = allCuisines
        .filter(c => c.includes(query) && !selectedCuisines.includes(c))
        .slice(0, 10);

    if (matches.length === 0) {
        cuisineDrop.classList.remove("show");
        return;
    }

    cuisineDrop.innerHTML = matches
        .map(c => `<div class="cuisine-option" data-cuisine="${c}">${capitalise(c)}</div>`)
        .join("");
    cuisineDrop.classList.add("show");
});

cuisineInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        const val = cuisineInput.value.trim().toLowerCase();
        if (val && allCuisines.includes(val) && !selectedCuisines.includes(val)) {
            addCuisine(val);
        } else if (val) {
            // Check if there's a dropdown option we can pick
            const first = cuisineDrop.querySelector(".cuisine-option");
            if (first) addCuisine(first.dataset.cuisine);
        }
    }
});

cuisineDrop.addEventListener("click", (e) => {
    const opt = e.target.closest(".cuisine-option");
    if (opt) addCuisine(opt.dataset.cuisine);
});

// Close dropdown on outside click
document.addEventListener("click", (e) => {
    if (!e.target.closest(".cuisine-input-area")) {
        cuisineDrop.classList.remove("show");
    }
});

function addCuisine(cuisine) {
    if (selectedCuisines.includes(cuisine)) return;
    selectedCuisines.push(cuisine);
    renderCuisineTags();
    cuisineInput.value = "";
    cuisineDrop.classList.remove("show");
}

function removeCuisine(cuisine) {
    selectedCuisines = selectedCuisines.filter(c => c !== cuisine);
    renderCuisineTags();
}

function renderCuisineTags() {
    selectedBox.innerHTML = selectedCuisines
        .map(c => `
            <span class="cuisine-tag">
                ${capitalise(c)}
                <span class="remove-tag" data-cuisine="${c}">×</span>
            </span>
        `).join("");

    selectedBox.querySelectorAll(".remove-tag").forEach(btn => {
        btn.addEventListener("click", () => removeCuisine(btn.dataset.cuisine));
    });
}

// ================================================================
// Rating slider
// ================================================================
ratingSlider.addEventListener("input", () => {
    ratingValue.textContent = parseFloat(ratingSlider.value).toFixed(1);
});

// ================================================================
// Form submission (with 429 handling + cooldown)
// ================================================================
let _cooldownTimer = null;

function startCooldown(seconds) {
    submitBtn.disabled = true;
    submitBtn.classList.remove("loading");
    let remaining = seconds;

    const originalText = submitBtn.querySelector(".btn-text");
    const savedText = originalText ? originalText.textContent : "";

    function tick() {
        if (originalText) {
            originalText.textContent = `Wait ${remaining}s...`;
        }
        remaining--;
        if (remaining < 0) {
            clearInterval(_cooldownTimer);
            _cooldownTimer = null;
            submitBtn.disabled = false;
            if (originalText) originalText.textContent = savedText;
        }
    }

    tick();
    _cooldownTimer = setInterval(tick, 1000);
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError();

    // Block if cooldown is active
    if (_cooldownTimer) return;

    // Gather form data
    const payload = {
        location: locationSel.value,
        budget: document.querySelector('input[name="budget"]:checked')?.value || "medium",
        cuisines: selectedCuisines,
        min_rating: parseFloat(ratingSlider.value),
        additional_preferences: document.getElementById("additional").value.trim(),
    };

    // Validation
    if (!payload.location) {
        showError("Please select a city.");
        return;
    }

    // Set loading state
    submitBtn.classList.add("loading");
    submitBtn.disabled = true;

    try {
        const res = await fetch("/api/recommend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const data = await res.json();

        // Handle rate limiting (429)
        if (res.status === 429) {
            const retryAfter = data.retry_after || 10;
            showError(data.error || `Too many requests. Please wait ${retryAfter} seconds.`);
            startCooldown(retryAfter);
            return;
        }

        if (!res.ok || !data.success) {
            showError(data.error || "Something went wrong. Please try again.");
            return;
        }

        displayResults(data);

    } catch (err) {
        console.error(err);
        showError("Network error. Please check if the server is running.");
    } finally {
        if (!_cooldownTimer) {
            submitBtn.classList.remove("loading");
            submitBtn.disabled = false;
        }
    }
});

// ================================================================
// Results display
// ================================================================
const MEDALS = { 1: "🥇", 2: "🥈", 3: "🥉" };

function displayResults(data) {
    const { recommendations, filters_applied, count } = data;

    // Update header
    resultsTitle.textContent = `${count} Restaurant${count !== 1 ? "s" : ""} Found`;
    resultsSub.textContent = `Showing results for ${capitalise(filters_applied.location)} · ${capitalise(filters_applied.budget)} budget`;

    // Build cards
    resultsGrid.innerHTML = recommendations.map(rec => {
        const medal = MEDALS[rec.rank] || `#${rec.rank}`;
        const ratingClass = rec.rating >= 4 ? "rating-high" : rec.rating >= 3 ? "rating-mid" : "rating-low";
        const cuisinesStr = Array.isArray(rec.cuisines) ? rec.cuisines.map(capitalise).join(", ") : rec.cuisines;
        const costFormatted = `₹${Number(rec.cost_for_two).toLocaleString("en-IN")}`;

        return `
            <article class="rec-card" id="rec-${rec.rank}">
                <div class="rec-card-header">
                    <span class="rec-rank">${medal}</span>
                    <h3 class="rec-name">${escapeHTML(rec.restaurant_name)}</h3>
                    <span class="rec-rating-badge ${ratingClass}">
                        ⭐ ${rec.rating.toFixed(1)}
                    </span>
                </div>
                <div class="rec-meta">
                    <span class="meta-item"><span class="meta-icon">🍕</span> ${escapeHTML(cuisinesStr)}</span>
                    <span class="meta-item"><span class="meta-icon">💰</span> ${costFormatted} for two</span>
                    <span class="meta-item"><span class="meta-icon">📍</span> ${capitalise(rec.location)}</span>
                </div>
                <div class="rec-explanation">
                    <span class="explain-label">🤖 Why this restaurant?</span>
                    ${escapeHTML(rec.explanation)}
                </div>
                ${rec.trade_offs ? `
                    <div class="rec-trade-offs">
                        <span>⚠️</span>
                        <span>${escapeHTML(rec.trade_offs)}</span>
                    </div>
                ` : ""}
            </article>
        `;
    }).join("");

    // Switch views
    formSection.classList.add("hidden");
    resultsSection.classList.remove("hidden");
    window.scrollTo({ top: 0, behavior: "smooth" });
}

// ================================================================
// Back button
// ================================================================
backBtn.addEventListener("click", () => {
    resultsSection.classList.add("hidden");
    formSection.classList.remove("hidden");
    formSection.style.animation = "none";
    requestAnimationFrame(() => {
        formSection.style.animation = "";
    });
});

// ================================================================
// Error handling
// ================================================================
function showError(message) {
    errorMsg.textContent = message;
    errorToast.classList.remove("hidden");
    setTimeout(hideError, 8000);
}

function hideError() {
    errorToast.classList.add("hidden");
}

errorClose.addEventListener("click", hideError);

// ================================================================
// Utilities
// ================================================================
function capitalise(str) {
    if (!str) return "";
    return str.replace(/\b\w/g, c => c.toUpperCase());
}

function escapeHTML(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// ================================================================
// Boot
// ================================================================
init();
