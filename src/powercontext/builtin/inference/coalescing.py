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

"""Share in-flight embedding work and reuse successful vectors across calls."""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from collections.abc import Sequence

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


class _Slot:
    """One text shared by every caller waiting on the same provider batch."""

    def __init__(self) -> None:
        self.future: asyncio.Future[EmbeddingVector] = asyncio.get_running_loop().create_future()
        self.usage: InferenceUsage | None = None
        self.future.add_done_callback(_retrieve_exception)


class CoalescingEmbeddingModel:
    """Batch identical texts, isolate cancellation, and keep a bounded vector cache."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if isinstance(coalesce_ms, bool) or not isinstance(coalesce_ms, (int, float)) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number of milliseconds")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._slots: dict[str, _Slot] = {}
        self._pending: deque[str] = deque()
        self._flushing = False

    @property
    def profile(self):
        return self._model.profile

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return one vector per input text, including duplicates, in input order."""

        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        slots: list[_Slot | None] = []
        cached: list[EmbeddingVector | None] = []
        for text in texts:
            hit = self._lookup(text)
            if hit is not None:
                slots.append(None)
                cached.append(hit)
                continue
            slots.append(self._reserve(text))
            cached.append(None)
        waited = [slot for slot in slots if slot is not None]
        if waited:
            await asyncio.gather(*(asyncio.shield(slot.future) for slot in waited))
        vectors = tuple(hit if slot is None else slot.future.result() for slot, hit in zip(slots, cached, strict=True))
        return EmbeddingResult(vectors=vectors, usage=_combine(waited))

    def _lookup(self, text: str) -> EmbeddingVector | None:
        if self._cache_size <= 0:
            return None
        vector = self._cache.get(text)
        if vector is None:
            return None
        self._cache.move_to_end(text)
        return vector

    def _reserve(self, text: str) -> _Slot:
        slot = self._slots.get(text)
        if slot is not None:
            return slot
        slot = _Slot()
        self._slots[text] = slot
        self._pending.append(text)
        self._ensure_flush()
        return slot

    def _ensure_flush(self) -> None:
        if self._flushing:
            return
        self._flushing = True
        asyncio.get_running_loop().create_task(self._drain())

    async def _drain(self) -> None:
        try:
            # Yield even for a zero window so callers already queued on this loop can join.
            await asyncio.sleep(self._coalesce_seconds)
            while self._pending:
                batch: list[str] = []
                while self._pending and len(batch) < self._max_batch_size:
                    batch.append(self._pending.popleft())
                await self._execute(batch)
        finally:
            self._flushing = False
            if self._pending:
                self._ensure_flush()

    async def _execute(self, texts: list[str]) -> None:
        try:
            result = await self._model.embed(tuple(texts))
        except Exception as error:
            self._reject(texts, error)
            return
        vectors = result.vectors
        if len(vectors) != len(texts):
            self._reject(texts, ValueError("embedding vector count does not match the batch"))
            return
        for text, vector in zip(texts, vectors, strict=True):
            normalized = tuple(float(item) for item in vector)
            slot = self._slots.pop(text, None)
            if self._cache_size > 0:
                self._cache[text] = normalized
                self._cache.move_to_end(text)
            if slot is not None:
                slot.usage = result.usage
                if not slot.future.done():
                    slot.future.set_result(normalized)
        while self._cache_size > 0 and len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _reject(self, texts: Sequence[str], error: BaseException) -> None:
        current: BaseException = error
        for text in texts:
            slot = self._slots.pop(text, None)
            if slot is None or slot.future.done():
                continue
            slot.future.set_exception(current)
            current = _fresh_exception(current)


def _combine(slots: Sequence[_Slot]) -> InferenceUsage:
    seen: set[int] = set()
    usages: list[InferenceUsage] = []
    for slot in slots:
        usage = slot.usage
        if usage is None or id(usage) in seen:
            continue
        seen.add(id(usage))
        usages.append(usage)
    if not usages:
        return InferenceUsage(requests=0)
    return InferenceUsage(
        requests=sum(item.requests for item in usages),
        input_tokens=_total(item.input_tokens for item in usages),
        output_tokens=_total(item.output_tokens for item in usages),
    )


def _total(values) -> int | None:
    items = tuple(values)
    if all(item is None for item in items):
        return None
    return sum(item or 0 for item in items)


def _fresh_exception(error: BaseException) -> BaseException:
    try:
        return type(error)(*error.args) if error.args else type(error)()
    except Exception:
        return error


def _retrieve_exception(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()


__all__ = ["CoalescingEmbeddingModel"]
