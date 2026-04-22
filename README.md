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
├── .env                     # Local environment variables (API keys)
├── Procfile                 # Cloud Run/Render process file (gunicorn config)
├── Dockerfile               # Google Cloud Run deployment container config
├── render.yaml              # Render platform one-click deployment blueprint
├── runtime.txt              # Cloud Python version specification (3.11.9)
├── requirements.txt         # Pinned dependencies (Flask, Pandas, PyArrow, etc.)
│
├── static/                  # Web frontend assets
│   ├── index.html           # Main page UI
│   ├── style.css            # Dark glassmorphism design system
│   └── app.js               # Frontend logic (form handling, rate limiting)
│
├── data_ingestion/          # Phase 1: Data pipeline
│   ├── data_loader.py       # Multi-strategy Hugging Face loader (Parquet/CSV)
│   ├── data_cleaner.py      # Schema validation and data normalization
│   └── data_store.py        # Pickle caching mechanism
│
├── user_input/              # Phase 2: User preferences
│   ├── input_collector.py   # CLI interactive collection
│   └── input_validator.py   # Data validation and constraints
│
├── filtering/               # Phase 3: Selection logic
│   ├── filter_engine.py     # Hard-filtering criteria (budget, location, etc.)
│   ├── shortlister.py       # Diversity-aware sampling
│   ├── prompt_builder.py    # Assembling candidates into LLM context
│   └── prompt_templates.py  # System and user instructions
│
├── llm/                     # Phase 4: Recommendation Engine
│   ├── llm_client.py        # Gemini/OpenAI API client with token bucket rate limiting
│   ├── response_parser.py   # Robust JSON extraction and truncation recovery
│   └── fallback.py          # Heuristic ranking (when LLM goes offline)
│
├── output/                  # Phase 5: Presentation
│   ├── formatter.py         # Standardized data structuring
│   ├── cli_renderer.py      # Rich terminal UI cards
│   └── web_renderer.py      # Web API JSON packaging
│
├── tests/                   # Unit & integration tests
└── prompts/                 # Versioned text prompt templates (e.g., v1.txt)
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
