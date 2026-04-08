"""
queue_io.py – Redis Stream reader and writer wrappers.
"""

import time

import redis
from loguru import logger


class RedisStreamReader:
    """
    Reads messages from a Redis Stream using consumer groups (XREADGROUP).
    Creates the stream and consumer group on first use if they don't exist.
    Acknowledges processed messages with XACK.
    """

    def __init__(
        self,
        redis_url: str,
        stream: str,
        group: str,
        consumer: str,
        batch_size: int = 8,
        block_ms: int = 50,
    ):
        self._url = redis_url
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._batch_size = batch_size
        self._block_ms = block_ms
        self._client: redis.Redis | None = None
        self._connect()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self):
        retries = 0
        while True:
            try:
                self._client = redis.from_url(self._url, decode_responses=False)
                self._client.ping()
                self._ensure_group()
                logger.info(f"RedisStreamReader connected → {self._url} | stream={self._stream}")
                return
            except Exception as exc:
                wait = min(2 ** retries, 30)
                logger.warning(f"Redis connect failed ({exc}). Retry in {wait}s…")
                time.sleep(wait)
                retries += 1

    def _ensure_group(self):
        """Create stream + consumer group if they don't already exist."""
        try:
            self._client.xgroup_create(
                self._stream, self._group, id="0", mkstream=True
            )
            logger.info(f"Consumer group '{self._group}' created on stream '{self._stream}'.")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                pass  # group already exists — fine
            else:
                raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> list[tuple[bytes, dict]]:
        """
        Read up to batch_size pending messages.
        Returns list of (message_id_bytes, fields_dict) pairs.
        fields_dict values are raw bytes.
        """
        while True:
            try:
                raw = self._client.xreadgroup(
                    groupname=self._group,
                    consumername=self._consumer,
                    streams={self._stream: ">"},
                    count=self._batch_size,
                    block=self._block_ms,
                )
                if not raw:
                    return []
                # raw = [(stream_name, [(msg_id, {field: value}), ...])]
                _, messages = raw[0]
                return messages
            except redis.exceptions.ConnectionError as exc:
                logger.warning(f"Redis read error ({exc}). Reconnecting…")
                self._connect()

    def ack(self, message_ids: list[bytes]):
        """Acknowledge a list of message IDs."""
        if not message_ids:
            return
        try:
            self._client.xack(self._stream, self._group, *message_ids)
        except redis.exceptions.ConnectionError as exc:
            logger.warning(f"Redis ACK error ({exc}). Messages may be reprocessed.")


class RedisStreamWriter:
    """Writes dicts to a Redis Stream with XADD."""

    def __init__(self, redis_url: str, stream: str, maxlen: int | None = 10_000):
        self._url = redis_url
        self._stream = stream
        self._maxlen = maxlen
        self._client: redis.Redis | None = None
        self._connect()

    def _connect(self):
        retries = 0
        while True:
            try:
                self._client = redis.from_url(self._url, decode_responses=False)
                self._client.ping()
                logger.info(f"RedisStreamWriter connected → {self._url} | stream={self._stream}")
                return
            except Exception as exc:
                wait = min(2 ** retries, 30)
                logger.warning(f"Redis write-connect failed ({exc}). Retry in {wait}s…")
                time.sleep(wait)
                retries += 1

    def write(self, fields: dict):
        """
        Write a message to the stream, trimming to maxlen with ~ approximation
        for efficiency. Retries on connection error.
        """
        while True:
            try:
                if self._maxlen:
                    self._client.xadd(self._stream, fields, maxlen=self._maxlen, approximate=True)
                else:
                    self._client.xadd(self._stream, fields)
                return
            except redis.exceptions.ConnectionError as exc:
                logger.warning(f"Redis write error ({exc}). Reconnecting…")
                self._connect()
