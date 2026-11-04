"""Test live running server on http://127.0.0.1:8100."""

import json
import time
import urllib.request

base_url = "http://127.0.0.1:8100"

print("=================================================================")
print("1. HEALTH CHECK (GET /health)")
print("=================================================================")
with urllib.request.urlopen(f"{base_url}/health") as r:
    data = json.loads(r.read().decode("utf-8"))
    print(json.dumps(data, indent=2))


def call_recommend(payload: dict, label: str):
    print("\n" + "=" * 65)
    print(label)
    print("=" * 65)
    req = urllib.request.Request(
        f"{base_url}/recommend",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        resp = json.loads(r.read().decode("utf-8"))
        wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        print(f"Server Latency: {resp.get('latency_ms')} ms (Round-trip: {wall_ms} ms)")
        print(f"Source: {resp.get('source')} | Cache Hit: {resp.get('cache_hit')} | Variant: {resp.get('variant')}")
        print("Recommendations:")
        for item in resp.get("items", []):
            print(f"  #{item['rank']} [{item['score']:.4f}] {item['title']} ({item['genres']})")


# 2. Warm user with Variant B (Two-Stage)
call_recommend(
    {"user_id": 105, "k": 5, "variant": "B"},
    "2. WARM USER (User 105: Sci-Fi Fan) -> VARIANT B (Two-Stage Engine)",
)

# 3. Warm user with Variant A (Retrieval Only)
call_recommend(
    {"user_id": 105, "k": 5, "variant": "A"},
    "3. WARM USER (User 105: Sci-Fi Fan) -> VARIANT A (Stage 1 Retrieval Only)",
)

# 4. Cache hit test (repeated call for Variant B)
call_recommend(
    {"user_id": 105, "k": 5, "variant": "B"},
    "4. CACHE HIT TEST (Repeated Call for User 105 Variant B)",
)

# 5. Cold-start new user
call_recommend(
    {"user_id": 999999, "k": 5, "seed_genres": ["Sci-Fi", "Action"]},
    "5. COLD-START NEW USER (User 999999, Seed Genres: Sci-Fi + Action)",
)

# 6. Cold-start new user with Drama/Crime
call_recommend(
    {"user_id": 888888, "k": 5, "seed_genres": ["Drama", "Crime"]},
    "6. COLD-START NEW USER (User 888888, Seed Genres: Drama + Crime)",
)

