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

"""HTTP score reranking that keeps original Memory candidate identity."""

from __future__ import annotations

import math
from json import JSONDecodeError

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Score candidates through an HTTP service and select them globally."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Request every original rank once, then keep the best global scores."""

        if len(candidates) < 1:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if type(limit) is not int or limit < 1 or limit > len(candidates):
            raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003
        scores: dict[int, float] = {}
        for start in range(0, len(candidates), self._batch_size):
            batch = tuple(enumerate(candidates[start : start + self._batch_size], start=start + 1))
            scores.update(await self._score_batch(query, batch))
        selected = tuple(
            rank
            for rank, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        )
        return MemoryRerankDecision(
            selected_ranks=selected,
            usage=InferenceUsage(requests=math.ceil(len(candidates) / self._batch_size)),
        )

    async def _score_batch(self, query: str, batch: tuple[tuple[int, MemoryHit], ...]) -> dict[int, float]:
        expected = {rank for rank, _candidate in batch}
        response = await self._client.post(
            self._endpoint,
            json={
                "query": query,
                "candidates": [{"rank": rank, "text": candidate.text} for rank, candidate in batch],
            },
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except JSONDecodeError as error:
            raise ValueError("rerank response is not JSON") from error  # noqa: TRY003
        return _scores(payload, expected)


def _scores(payload: object, expected: set[int]) -> dict[int, float]:
    if not isinstance(payload, dict) or set(payload) != {"scores"} or not isinstance(payload["scores"], list):
        raise ValueError("rerank response schema is invalid")  # noqa: TRY003
    found: dict[int, float] = {}
    for item in payload["scores"]:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("rerank score schema is invalid")  # noqa: TRY003
        rank = item["rank"]
        score = item["score"]
        if type(rank) is not int or rank not in expected or rank in found:
            raise ValueError("rerank score rank is invalid")  # noqa: TRY003
        if isinstance(score, bool) or type(score) not in {int, float} or not math.isfinite(score):
            raise ValueError("rerank score must be finite")  # noqa: TRY003
        found[rank] = float(score)
    if set(found) != expected:
        raise ValueError("rerank scores must cover the batch once")  # noqa: TRY003
    return found
