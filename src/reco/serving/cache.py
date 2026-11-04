"""
Redis caching layer for recommendation lists, popularity lists, and user features.
Includes key versioning, configurable TTLs, and graceful failure on Redis disconnect.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import redis

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis cache wrapper supporting:
    - Versioned recommendation lists: rec:{variant}:{user_id}:{k} (TTL 3600s)
    - Global popularity: pop:global (TTL 86400s)
    - Per-genre popularity: pop:genre:{genre} (TTL 86400s)
    - Hot user features: ufeat:{user_id} (TTL 21600s)
    - Graceful degradation: all operations catch Redis errors, log warnings, and fail open.
    """

    DEFAULT_REC_TTL = 3600  # 1 hour
    DEFAULT_POP_TTL = 86400  # 24 hours
    DEFAULT_UFEAT_TTL = 21600  # 6 hours

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: float = 1.0,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.host = host
        self.port = port
        self.db = db
        self.client: Optional[redis.Redis] = None
        if enabled:
            try:
                self.client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    socket_timeout=socket_timeout,
                    socket_connect_timeout=socket_timeout,
                    decode_responses=True,
                )
            except Exception as e:
                logger.warning("Failed to initialize Redis client (%s:%d): %s", host, port, e)
                self.client = None

    def is_available(self) -> bool:
        """Check if Redis connection is active."""
        if not self.enabled or self.client is None:
            return False
        try:
            return bool(self.client.ping())
        except Exception as e:
            logger.warning("Redis ping failed (%s:%d): %s", self.host, self.port, e)
            return False

    # --- Recommendation Lists ---

    @staticmethod
    def format_rec_key(variant: str, user_id: int, k: int) -> str:
        """Generate standardized recommendation cache key."""
        return f"rec:{variant}:{user_id}:{k}"

    def get_recommendations(
        self, variant: str, user_id: int, k: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached recommendation list for (variant, user_id, k). Returns None on miss/error."""
        if not self.enabled or self.client is None:
            return None
        key = self.format_rec_key(variant, user_id, k)
        try:
            val = self.client.get(key)
            if val is not None:
                return json.loads(val)
        except Exception as e:
            logger.warning("Error reading from Redis key %s: %s", key, e)
        return None

    def set_recommendations(
        self,
        variant: str,
        user_id: int,
        k: int,
        recommendations: List[Dict[str, Any]],
        ttl: int = DEFAULT_REC_TTL,
    ) -> bool:
        """Store recommendation list with TTL. Returns True on success, False on error."""
        if not self.enabled or self.client is None:
            return False
        key = self.format_rec_key(variant, user_id, k)
        try:
            self.client.set(key, json.dumps(recommendations), ex=ttl)
            return True
        except Exception as e:
            logger.warning("Error writing to Redis key %s: %s", key, e)
            return False

    # --- Popularity Lists ---

    def get_global_popularity(self, k: int = 10) -> Optional[List[int]]:
        """Retrieve cached global popularity list."""
        if not self.enabled or self.client is None:
            return None
        key = "pop:global"
        try:
            val = self.client.get(key)
            if val is not None:
                items = json.loads(val)
                return items[:k]
        except Exception as e:
            logger.warning("Error reading global popularity from Redis: %s", e)
        return None

    def set_global_popularity(
        self,
        movie_ids: List[int],
        ttl: int = DEFAULT_POP_TTL,
    ) -> bool:
        """Cache global popularity list with TTL."""
        if not self.enabled or self.client is None:
            return False
        key = "pop:global"
        try:
            self.client.set(key, json.dumps(movie_ids), ex=ttl)
            return True
        except Exception as e:
            logger.warning("Error writing global popularity to Redis: %s", e)
            return False

    def get_genre_popularity(self, genre: str, k: int = 10) -> Optional[List[int]]:
        """Retrieve cached per-genre popularity list."""
        if not self.enabled or self.client is None:
            return None
        key = f"pop:genre:{genre}"
        try:
            val = self.client.get(key)
            if val is not None:
                items = json.loads(val)
                return items[:k]
        except Exception as e:
            logger.warning("Error reading genre popularity for %s from Redis: %s", genre, e)
        return None

    def set_genre_popularity(
        self,
        genre: str,
        movie_ids: List[int],
        ttl: int = DEFAULT_POP_TTL,
    ) -> bool:
        """Cache per-genre popularity list with TTL."""
        if not self.enabled or self.client is None:
            return False
        key = f"pop:genre:{genre}"
        try:
            self.client.set(key, json.dumps(movie_ids), ex=ttl)
            return True
        except Exception as e:
            logger.warning("Error writing genre popularity for %s to Redis: %s", genre, e)
            return False

    # --- User Features ---

    def get_user_features(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve cached user features."""
        if not self.enabled or self.client is None:
            return None
        key = f"ufeat:{user_id}"
        try:
            val = self.client.get(key)
            if val is not None:
                return json.loads(val)
        except Exception as e:
            logger.warning("Error reading user features for %d from Redis: %s", user_id, e)
        return None

    def set_user_features(
        self,
        user_id: int,
        features: Dict[str, Any],
        ttl: int = DEFAULT_UFEAT_TTL,
    ) -> bool:
        """Cache user features with TTL."""
        if not self.enabled or self.client is None:
            return False
        key = f"ufeat:{user_id}"
        try:
            self.client.set(key, json.dumps(features), ex=ttl)
            return True
        except Exception as e:
            logger.warning("Error writing user features for %d to Redis: %s", user_id, e)
            return False

    # --- Cache Management ---

    def flush_variant(self, variant: str) -> int:
        """Flush all recommendation keys for a given model variant / version (e.g. 'v1')."""
        if not self.enabled or self.client is None:
            return 0
        pattern = f"rec:{variant}:*"
        count = 0
        try:
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    self.client.delete(*keys)
                    count += len(keys)
                if cursor == 0:
                    break
            logger.info("Flushed %d keys matching pattern %s", count, pattern)
        except Exception as e:
            logger.warning("Error flushing variant %s: %s", variant, e)
        return count

    def get_ttl(self, key: str) -> int:
        """Return the TTL of a key in seconds (-1 if no TTL, -2 if not found, -3 on error)."""
        if not self.enabled or self.client is None:
            return -3
        try:
            return self.client.ttl(key)
        except Exception as e:
            logger.warning("Error checking TTL for %s: %s", key, e)
            return -3
