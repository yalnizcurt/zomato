# 🍽️ Restaurant Recommender — AI-Powered

An AI-powered restaurant recommendation system inspired by Zomato. It ingests real restaurant data, takes your preferences, and uses an LLM to generate personalized, explainable recommendations.

## Quick Start

### 1. Clone & setup
```bash
cd restaurant-recommender
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
# Edit .env with your actual API keys
```

### 3. Run (Web UI — recommended)
```bash
python app.py
# Open http://localhost:5000 in your browser
```

### 3b. Run (CLI mode)
```bash
python main.py
```

## Project Structure

```
restaurant-recommender/
├── app.py                   # 🌐 Web UI entry point (Flask) — recommended
├── main.py                  # CLI entry point — orchestrates all 5 phases
├── config.py                # App-wide configuration
├── models.py                # Shared data models & exceptions
├── static/                  # Web frontend
│   ├── index.html           # Main page
│   ├── style.css            # Dark glassmorphism design system
│   └── app.js               # Frontend logic (form, API calls, rendering)
├── phase1_data/             # Data ingestion & preprocessing
├── phase2_input/            # User input collection & validation
├── phase3_integration/      # Filtering, shortlisting, prompt building
├── phase4_llm/              # LLM client, response parsing, fallback
├── phase5_output/           # Output formatting & rendering
├── tests/                   # Unit & integration tests
├── prompts/                 # Versioned prompt templates
└── requirements.txt         # Pinned dependencies
```

## Architecture

See [`architecture.md`](../architecture.md) for full phase-wise architecture with diagrams.

## Configuration

| Variable | Default | Description |
|:---------|:--------|:------------|
| `LLM_PROVIDER` | `gemini` | `openai` or `gemini` |
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `GEMINI_API_KEY` | — | Your Google Gemini API key |
| `APP_ENV` | `development` | `development` or `production` |
| `MAX_CANDIDATES` | `15` | Max restaurants sent to LLM |
| `MAX_RECOMMENDATIONS` | `5` | Number of recommendations returned |

## Testing
```bash
pytest tests/ -v --cov
```

## License

MIT
