# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prefix-reusable context packs and their incremental deltas."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})


class ContextPackError(Exception):
    """A context pack or delta failed protocol or chunk validation."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete context payload split into checked UTF-8 chunks."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...] | Sequence[bytes]) -> ContextPack:
        """Build a pack from chunks that are already the intended wire pieces."""

        if isinstance(items, (str, bytes)) or any(not isinstance(item, bytes) for item in items):
            raise ContextPackError("pack items must be bytes")
        encoded = [_encode(item) for item in items]
        digest = _digest(encoded)
        raw = _pack_bytes(encoded, digest)
        return cls(items=tuple(items), digest=digest, byte_count=len(raw))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = 512) -> ContextPack:
        """Split text on deterministic UTF-8 boundaries that respect the byte limit."""

        if not isinstance(content, str):
            raise ContextPackError("content must be text")
        return cls.build(_chunk(content, chunk_bytes))

    def to_bytes(self) -> bytes:
        """Serialize this pack in canonical JSON."""

        return _pack_bytes([_encode(item) for item in self.items], self.digest)

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        """Decode a pack and reject a digest, size, or schema mismatch."""

        payload = _object(data, _PACK_FIELDS)
        if payload.get("schema") != _PACK_SCHEMA:
            raise ContextPackError("pack schema is invalid")
        byte_count = payload["byte_count"]
        if type(byte_count) is not int or byte_count != len(data):
            raise ContextPackError("pack byte count is invalid")
        encoded = _encoded_items(payload["items"])
        digest = payload["digest"]
        if digest != _digest(encoded):
            raise ContextPackError("pack digest is invalid")
        if _pack_bytes(encoded, digest) != data:
            raise ContextPackError("pack encoding is invalid")
        return cls(items=tuple(_decode(value) for value in encoded), digest=digest, byte_count=byte_count)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """Instructions for rebuilding a target pack from a shared base prefix."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Keep the longest chunk prefix that both packs already share."""

        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("delta requires two packs")
        prefix = 0
        limit = min(len(base.items), len(target.items))
        while prefix < limit and base.items[prefix] == target.items[prefix]:
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize this delta in canonical JSON without its byte count."""

        _require_prefix(self.prefix_items)
        return _dumps(
            {
                "schema": _DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": [_encode(item) for item in self.appended],
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        """Decode a delta and reject a schema, prefix, or item encoding error."""

        payload = _object(data, _DELTA_FIELDS)
        if payload.get("schema") != _DELTA_SCHEMA:
            raise ContextPackError("delta schema is invalid")
        _require_prefix(payload["prefix_items"])
        if not isinstance(payload["base_digest"], str) or not isinstance(payload["target_digest"], str):
            raise ContextPackError("delta digest is invalid")
        appended = tuple(_decode(value) for value in _encoded_items(payload["appended"]))
        return cls(payload["base_digest"], payload["target_digest"], payload["prefix_items"], appended)


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from a base pack and a validated delta."""

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta cannot be applied")
    _require_prefix(delta.prefix_items)
    if delta.prefix_items > len(base.items) or base.digest != delta.base_digest:
        raise ContextPackError("delta does not match the base pack")
    target = ContextPack.build((*base.items[: delta.prefix_items], *delta.appended))
    if target.digest != delta.target_digest:
        raise ContextPackError("delta target digest is invalid")
    return target


def _chunk(content: str, chunk_bytes: int) -> tuple[bytes, ...]:
    if type(chunk_bytes) is not int or chunk_bytes < 1:
        raise ContextPackError("chunk size must be a positive integer")
    data = content.encode("utf-8")
    items: list[bytes] = []
    start = 0
    while start < len(data):
        end = min(start + chunk_bytes, len(data))
        while end > start and end < len(data) and data[end] & 0xC0 == 0x80:
            end -= 1
        if end == start:
            raise ContextPackError("chunk size cannot hold a UTF-8 character")
        items.append(data[start:end])
        start = end
    return tuple(items)


def _digest(items: list[str]) -> str:
    raw = _dumps({"schema": _PACK_SCHEMA, "items": items})
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _pack_bytes(items: list[str], digest: str) -> bytes:
    body = {"schema": _PACK_SCHEMA, "items": items, "digest": digest}
    guess = 0
    encoded = b""
    for _ in range(8):
        encoded = _dumps({**body, "byte_count": guess})
        if len(encoded) == guess:
            return encoded
        guess = len(encoded)
    raise ContextPackError("pack byte count did not converge")


def _dumps(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _object(data: bytes, fields: frozenset[str]) -> dict[str, object]:
    if not isinstance(data, bytes):
        raise ContextPackError("pack bytes are invalid")
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise ContextPackError("pack JSON is invalid") from error
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ContextPackError("pack fields are invalid")
    return payload


def _encoded_items(value: object) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ContextPackError("pack items are invalid")
    return value


def _encode(item: bytes) -> str:
    if not isinstance(item, bytes):
        raise ContextPackError("pack item must be bytes")
    return base64.b64encode(item).decode("ascii")


def _decode(value: str) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except Exception as error:
        raise ContextPackError("pack item encoding is invalid") from error
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ContextPackError("pack item encoding is invalid")
    return decoded


def _require_prefix(value: object) -> None:
    if type(value) is not int or value < 0:
        raise ContextPackError("prefix must be a non-negative integer")
