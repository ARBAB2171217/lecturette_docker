# SSB Lecturette Agent

Production-ready, cost-optimized, retrieval-first AI system that generates
3-minute SSB-style lecturettes for NDA / CDS / AFCAT / SSB Interview prep
(Defence Current Affairs, Geopolitics, IR, Social Issues, National Security,
Technology, Leadership, Psychology, Abstract Topics).

## Architecture

Two agents only:

1. **Research & Structure Agent** — turns retrieved/researched material into
   structured notes + a lecturette outline.
2. **Writer & Quality Agent** — turns the outline into the final spoken-style
   markdown lecturette.

Cost control comes from the **retrieval layer**, not the agents:

```
User Topic
   -> Topic Parser (Gemini, cheap call)
   -> Embedding (text-embedding-004)
   -> pgvector similarity search (top 5)
   -> similarity >= 0.80 ? reuse cached research : Google Search
   -> Research Agent -> Writer Agent -> store + return
```

- Cache hit: skips Google Search entirely, target latency **< 3s**.
- Cache miss: runs targeted Google Search queries, target latency **< 10s**.
- Near-duplicate research (similarity > 0.95) updates the existing row
  instead of creating a duplicate.

## Project Structure

```
app/
├── agents/
│   ├── research_agent.py     # Agent 1
│   └── writer_agent.py       # Agent 2
├── database/
│   ├── connection.py         # async engine/session, init_db()
│   ├── models.py             # Lecturette, ResearchCache, SearchLog
│   └── vector_store.py       # pgvector similarity queries
├── services/
│   ├── gemini_client.py      # shared Gemini text + embedding wrapper
│   ├── parser_service.py     # topic -> keywords/category/normalized query
│   ├── google_search_service.py # web search (only provider used)
│   ├── embedding_service.py  # text-embedding-004 wrapper
│   └── retrieval_service.py  # cache-first orchestration logic
├── api/
│   └── lecturette.py         # POST /generate-lecturette
├── schemas/
│   └── lecturette_schema.py  # Pydantic request/response models
├── config/
│   └── settings.py           # env-driven settings
└── main.py                   # FastAPI app + lifespan (init_db on startup)
```

## Setup (local, no Docker)

1. Install PostgreSQL 16+ with the `pgvector` extension available.
2. Create the database:
   ```bash
   createdb lecturette_db
   ```
3. Copy env file and fill in your keys:
   ```bash
   cp .env.example .env
   # edit .env: GEMINI_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID, DATABASE_URL
   ```
4. Install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
5. Run the app (tables + pgvector extension are created automatically on startup):
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
6. Open docs: http://localhost:8000/docs

## Setup (Docker — recommended)

```bash
cp .env.example .env
# edit .env with your GEMINI_API_KEY, GOOGLE_API_KEY, and GOOGLE_CSE_ID
docker compose up --build
```

This starts:
- `db` — Postgres 16 with pgvector pre-installed (`pgvector/pgvector:pg16` image)
- `api` — the FastAPI app on port 8000

## API

### `POST /api/v1/generate-lecturette`

Request:
```json
{ "topic": "Artificial Intelligence in Defence" }
```

Response:
```json
{
  "topic": "Artificial Intelligence in Defence",
  "category": "Technology",
  "lecturette": "# Topic\n\n...",
  "source": "web",
  "similarity_score": 0.42,
  "saved": true,
  "cache_hit": false
}
```

### `GET /health`
Returns app + DB connectivity status.

## Notes / Production Hardening Checklist

- Add an `idx_research_cache_topic_trgm` (pg_trgm) index if you also want
  fuzzy text-based dedup alongside vector similarity.
- Add request-level auth (API key / JWT) before exposing publicly — not
  included here since it wasn't specified.
- Add `slowapi` or a reverse-proxy rate limiter for the public endpoint.
- Consider Alembic migrations (`alembic.ini` not included — `init_db()`
  currently does `create_all`, fine for early-stage use, swap to proper
  migrations once schema stabilizes).
- The HNSW index parameters (`m=16, ef_construction=64`) are reasonable
  defaults — tune based on corpus size once you have real volume.
