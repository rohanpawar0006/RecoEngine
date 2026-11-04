# RecoEngine Serving API Documentation

The RecoEngine serving layer is a high-performance FastAPI service designed for low-latency multi-stage recommendation retrieval and ranking.

- **Base URL:** `http://localhost:8100` (or `http://localhost:8000`)
- **OpenAPI Specification:** `docs/openapi.json` (also interactive at `/docs` and `/redoc`)
- **Framework:** FastAPI + Pydantic v2 + Uvicorn
- **Cache Layer:** Redis (fail-open caching with automatic versioned key invalidation)
- **Serving Engines:** Triton Inference Server (production) with automatic local in-process SavedModel fallback

---

## 1. Endpoints Overview

| Method | Endpoint | Description | Cacheable | Target Latency |
|---|---|---|---|---|
| `GET` | `/health` | Service health, components, and catalog status | No | < 5 ms |
| `POST` | `/recommend` | Personalized or cold-start recommendations | Yes (TTL 3600s) | p50 < 20 ms |
| `POST` | `/events` | Event tracking for user clicks and impressions | No | < 5 ms |
| `GET` | `/ab/summary` | Real-time CTR and impression metrics by variant | No | < 5 ms |

---

## 2. Detailed API Specifications

### `GET /health`

Returns operational status, model repository version, catalog size, and component connectivity (Redis, FAISS, Triton).

#### Request
```http
GET /health HTTP/1.1
Host: localhost:8100
Accept: application/json
```

#### Response (200 OK)
```json
{
  "status": "healthy",
  "version": "v1",
  "redis_available": true,
  "triton_available": false,
  "served_by": "local",
  "faiss_loaded": true,
  "catalog_size": 34461
}
```

---

### `POST /recommend`

Generates personalized recommendations for warm users or cold-start fallback recommendations for new users.

#### Request Body Schema (`RecommendationRequest`)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `user_id` | `integer` | Yes | – | Unique user identifier (`1` to `137,883` for warm users, or any unseen ID for cold start). |
| `k` | `integer` | No | `10` | Number of recommendations to return (range: `1` to `100`). |
| `variant` | `string` | No | `"auto"` | Experiment variant: `"auto"` (salted MD5 50/50 hash), `"A"` (Stage 1 retrieval-only), `"B"` (Two-stage with DCN-V2 ranker). |
| `seed_genres` | `list[string]` | No | `[]` | Optional list of genres (e.g. `["Sci-Fi", "Action"]`) to guide content-based cold-start fallback. |

#### Request Example (Warm User Personalized)
```bash
curl -X POST "http://localhost:8100/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 105, "k": 10, "variant": "B"}'
```

#### Response (200 OK)
```json
{
  "user_id": 105,
  "variant": "B",
  "model_version": "v1",
  "source": "model_ranked",
  "cache_hit": false,
  "latency_ms": 17.65,
  "items": [
    {
      "item_id": 260,
      "title": "Star Wars: Episode IV - A New Hope (1977)",
      "genres": "Action|Adventure|Sci-Fi",
      "score": 0.9412,
      "rank": 1
    },
    {
      "item_id": 1196,
      "title": "Star Wars: Episode V - The Empire Strikes Back (1980)",
      "genres": "Action|Adventure|Sci-Fi",
      "score": 0.9385,
      "rank": 2
    }
  ]
}
```

#### Request Example (New User Cold-Start Quiz)
```bash
curl -X POST "http://localhost:8100/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 999999, "k": 5, "seed_genres": ["Comedy", "Romance"]}'
```

#### Response (200 OK)
```json
{
  "user_id": 999999,
  "variant": "A",
  "model_version": "v1",
  "source": "fallback_content_based",
  "cache_hit": false,
  "latency_ms": 6.82,
  "items": [
    {
      "item_id": 356,
      "title": "Forrest Gump (1994)",
      "genres": "Comedy|Drama|Romance|War",
      "score": 1.4821,
      "rank": 1
    }
  ]
}
```

---

### `POST /events`

Logs client engagement signals (impressions, clicks, ratings) for real-time telemetry and A/B tracking.

#### Request Body Schema (`EventPayload`)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `user_id` | `integer` | Yes | – | User identifier. |
| `item_id` | `integer` | Yes | – | Item identifier. |
| `event_type` | `string` | Yes | – | One of `"impression"`, `"click"`, `"rating"`. |
| `variant` | `string` | No | `"unknown"` | Experiment variant active when event occurred. |
| `timestamp` | `float` | No | Current time | Event timestamp (Unix epoch). |

#### Request Example
```bash
curl -X POST "http://localhost:8100/events" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 105, "item_id": 260, "event_type": "click", "variant": "B"}'
```

#### Response (200 OK)
```json
{
  "status": "success",
  "logged_event": {
    "user_id": 105,
    "item_id": 260,
    "event_type": "click",
    "variant": "B",
    "timestamp": 1793880000.0
  }
}
```

---

### `GET /ab/summary`

Returns aggregated impression and click statistics by experiment variant from the serving instance.

#### Request Example
```bash
curl -X GET "http://localhost:8100/ab/summary"
```

#### Response (200 OK)
```json
{
  "summary": {
    "A": {
      "impressions": 1500,
      "clicks": 65,
      "ctr": 0.0433
    },
    "B": {
      "impressions": 1520,
      "clicks": 76,
      "ctr": 0.0500
    }
  }
}
```

---

## 3. Error Handling & Resilience

The API implements strict validation and fail-open resilience patterns:

| Status Code | Scenario | Behavior |
|---|---|---|
| `422 Unprocessable Entity` | `k < 1` or `k > 100` | Rejected with detailed Pydantic validation error JSON. |
| `200 OK` (Degraded) | Redis connection timeout/down | Automatically bypassed (fail-open); serves via full compute pipeline without failing user request. |
| `200 OK` (Degraded) | Triton daemon unreachable | Automatically falls back to in-process TensorFlow SavedModel inference with zero user impact. |
| `200 OK` (Fallback) | User ID not found in training vocabulary | Automatically routes to time-decayed popularity or content-based seed matching fallback. |

