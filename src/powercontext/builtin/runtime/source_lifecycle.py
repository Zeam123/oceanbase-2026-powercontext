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

"""Propagate source availability through registered Memory and Topic Memory."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.artifacts.memory.errors import InvalidMemoryCitationError
from powercontext.builtin.artifacts.topic_memory import TopicMemory
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE
from powercontext.builtin.records import BaseAccessError
from powercontext.builtin.sources import validate_scope_id
from powercontext.sources import SourceRef

_UNREADABLE = (BaseAccessError, LookupError, InvalidMemoryCitationError)
_POLICIES = frozenset({"all", "any"})


class SourceLifecycleError(ValueError):
    """Raised when a lifecycle registration, event, or resolution is rejected."""


class SourceLifecycleEvent(BaseModel):
    """One availability change for an already registered source."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    node_id: str
    reason: str
    revision: int = Field(gt=0)
    sequence: int = Field(gt=0)
    available: bool

    @field_validator("node_id", "reason")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must contain a non-whitespace character")
        return value

    @field_validator("revision", "sequence")
    @classmethod
    def _reject_bool(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("must be a positive integer")
        return value

    @field_validator("available")
    @classmethod
    def _reject_non_bool(cls, value: bool) -> bool:
        if type(value) is not bool:
            raise ValueError("must be a boolean")
        return value


@dataclass
class _Node:
    node_id: str
    kind: str
    policy: str | None
    lifecycle_revision: int
    ref: SourceRef | MemoryCitation | ArtifactRef
    parents: tuple[str, ...]
    available: bool
    last_sequence: int
    last_event: dict[str, Any] | None
    ordinal: int


class SourceLifecycleApplication:
    """Expose one lifecycle service for each scope of a built-in runtime."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._locks: dict[str, asyncio.Lock] = {}

    def for_scope(self, scope_id: str) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self, validate_scope_id(scope_id))

    def _lock(self, scope_id: str) -> asyncio.Lock:
        lock = self._locks.get(scope_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[scope_id] = lock
        return lock


class ScopedSourceLifecycle:
    """Single-writer dependency graph for one scope."""

    def __init__(self, application: SourceLifecycleApplication, scope_id: str) -> None:
        self._application = application
        self._scope_id = scope_id

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int) -> None:
        node_id = _node_id(node_id)
        if not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("source-ref")
        revision = _positive_int(revision)
        await self._register(
            node_id,
            kind="source",
            policy=None,
            revision=revision,
            ref=source_ref,
            parents=(),
        )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: Sequence[str],
        policy: str,
    ) -> None:
        node_id = _node_id(node_id)
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("memory-ref")
        await self._register(
            node_id,
            kind="memory",
            policy=_policy(policy),
            revision=0,
            ref=citation,
            parents=_parents(parents),
        )

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        *,
        parents: Sequence[str],
        policy: str,
    ) -> None:
        node_id = _node_id(node_id)
        if not isinstance(artifact_ref, ArtifactRef) or artifact_ref.family != TopicMemory.family:
            raise SourceLifecycleError("topic-ref")
        await self._register(
            node_id,
            kind="topic",
            policy=_policy(policy),
            revision=0,
            ref=artifact_ref,
            parents=_parents(parents),
        )

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event")
        async with self._application._lock(self._scope_id):
            nodes = await self._load()
            source = nodes.get(event.node_id)
            if source is None or source.kind != "source":
                raise SourceLifecycleError("unknown-source")
            if event.revision != source.lifecycle_revision:
                raise SourceLifecycleError("revision")
            payload = event.model_dump(mode="json")
            if event.sequence == source.last_sequence:
                if source.last_event == payload:
                    return _active_ids(nodes)
                raise SourceLifecycleError("event-conflict")
            if event.sequence < source.last_sequence:
                raise SourceLifecycleError("sequence")
            before = {node.node_id: node.available for node in nodes.values()}
            source.available = event.available
            source.last_sequence = event.sequence
            source.last_event = payload
            _recompute(nodes)
            changed = [node for node in nodes.values() if node.available != before[node.node_id]]
            await self._commit(nodes, changed, new_node=None)
            return _active_ids(nodes)

    async def resolve(self, node_id: str, *, purpose: str) -> SourceRef | MemoryCitation | ArtifactRef:
        node_id = _node_id(node_id)
        if not isinstance(purpose, str) or purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("purpose")
        async with self._application._lock(self._scope_id):
            node = (await self._load()).get(node_id)
            if node is None:
                raise SourceLifecycleError("unknown-node")
            if purpose == "generation" and not node.available:
                raise SourceLifecycleError("unavailable")
            return node.ref

    async def _register(
        self,
        node_id: str,
        *,
        kind: str,
        policy: str | None,
        revision: int,
        ref: SourceRef | MemoryCitation | ArtifactRef,
        parents: tuple[str, ...],
    ) -> None:
        async with self._application._lock(self._scope_id):
            nodes = await self._load()
            current = nodes.get(node_id)
            if current is not None:
                if _same_binding(current, kind=kind, policy=policy, revision=revision, ref=ref, parents=parents):
                    return
                raise SourceLifecycleError("conflict")
            if kind != "source":
                _require_acyclic(nodes, node_id, parents)
            await self._require_readable(kind, ref)
            available = True if kind == "source" else _supported(nodes, parents, policy)
            node = _Node(
                node_id=node_id,
                kind=kind,
                policy=policy,
                lifecycle_revision=revision,
                ref=ref,
                parents=parents,
                available=available,
                last_sequence=0,
                last_event=None,
                ordinal=0,
            )
            await self._commit(nodes, [] if kind == "source" else [node], new_node=node)

    async def _require_readable(self, kind: str, ref: SourceRef | MemoryCitation | ArtifactRef) -> None:
        try:
            if isinstance(ref, SourceRef):
                await self._runtime.records.for_scope(self._scope_id).get_source(ref.source_type, ref.source_id)
                return
            if isinstance(ref, MemoryCitation):
                async with self._runtime._context(self._scope_id) as context:
                    await context.artifacts.memory.validate_citation(ref)
                return
            if isinstance(ref, ArtifactRef):
                await self._runtime.records.for_scope(self._scope_id).get_artifact_revision(
                    ref.family,
                    ref.artifact_id,
                    ref.revision,
                )
                return
        except _UNREADABLE as error:
            raise SourceLifecycleError("unreadable") from error
        raise SourceLifecycleError("unreadable")

    async def _commit(
        self,
        nodes: Mapping[str, _Node],
        changed: Sequence[_Node],
        *,
        new_node: _Node | None,
    ) -> None:
        # Native writes commit before the event row. A failed memory update can
        # therefore be retried as the same event instead of a completed replay.
        memories = [node for node in changed if node.kind == "memory"]
        topics = [node for node in changed if node.kind == "topic"]
        await self._sync_memories(memories)
        planned_topics: list[tuple[_Node, Any]] = []
        for node in topics:
            if not isinstance(node.ref, ArtifactRef):
                raise SourceLifecycleError("topic-ref")
            if await self._topic_visible(node.ref) == node.available:
                continue
            projection = await self._prepare_topic(node.ref) if node.available else None
            planned_topics.append((node, projection))
        async with self._database().transaction() as connection:
            repository = self._topics()
            for node, projection in planned_topics:
                await repository.set_searchable(
                    connection,
                    self._scope_id,
                    node.ref,
                    node.available,
                    projection=projection,
                )
            if new_node is None:
                await self._update_nodes(connection, nodes.values())
            else:
                await self._insert_node(connection, new_node)

    async def _sync_memories(self, nodes: Sequence[_Node]) -> None:
        if not nodes:
            return
        async with self._runtime._context(self._scope_id) as context:
            service = context.artifacts.memory
            current = await service.head(context.artifacts.memory_artifact_id)
            versions = {entry.entry_id: entry for entry in await service.entries(current)}
            manifest = {item.entry_id: item for item in current.content.manifest.entries}
            inactive: list[Any] = []
            active: list[Any] = []
            for node in nodes:
                citation = node.ref
                if not isinstance(citation, MemoryCitation):
                    raise SourceLifecycleError("memory-ref")
                item = manifest.get(citation.entry_id)
                version = versions.get(citation.entry_id)
                if item is None or version is None or version.entry_version_id != citation.entry_version_id:
                    raise SourceLifecycleError("memory-entry")
                if item.state == ("active" if node.available else "inactive"):
                    continue
                (active if node.available else inactive).append(version)
            if inactive:
                current = await service.forget(current, entries=tuple(inactive), reason="source-lifecycle")
            if active:
                if inactive:
                    refreshed = {entry.entry_id: entry for entry in await service.entries(current)}
                    active = [refreshed[entry.entry_id] for entry in active]
                await service.reactivate(current, entries=tuple(active), reason="source-lifecycle")

    async def _prepare_topic(self, ref: ArtifactRef) -> Any:
        async with self._database().transaction() as connection:
            topic = await self._topics().artifacts.get(connection, self._scope_id, ref)
        if not isinstance(topic, TopicMemory):
            raise SourceLifecycleError("topic")
        writer = self._runtime._record_service._family_writers.get(TopicMemory.family)
        return await writer.prepare(topic.content, usage_scope_id=self._scope_id)

    async def _topic_visible(self, ref: ArtifactRef) -> bool:
        async with self._database().transaction() as connection:
            return await self._topics()._is_searchable(connection, self._scope_id, ref.artifact_id)

    async def _load(self) -> dict[str, _Node]:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        async with self._database().transaction() as connection:
            rows = (
                await connection.execute(select(table).where(table.c.scope_id == self._scope_id).order_by(table.c.ordinal))
            ).mappings()
        nodes: dict[str, _Node] = {}
        for row in rows:
            kind = str(row["kind"])
            payload = json.loads(str(row["ref_json"]))
            event_json = row["last_event_json"]
            nodes[str(row["node_id"])] = _Node(
                node_id=str(row["node_id"]),
                kind=kind,
                policy=None if row["policy"] is None else str(row["policy"]),
                lifecycle_revision=int(row["lifecycle_revision"]),
                ref=_load_ref(kind, payload),
                parents=tuple(json.loads(str(row["parents_json"]))),
                available=bool(row["available"]),
                last_sequence=int(row["last_sequence"]),
                last_event=None if event_json is None else json.loads(str(event_json)),
                ordinal=int(row["ordinal"]),
            )
        return nodes

    async def _insert_node(self, connection: AsyncConnection, node: _Node) -> None:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        current = await connection.scalar(
            select(func.max(table.c.ordinal)).where(table.c.scope_id == self._scope_id)
        )
        node.ordinal = 0 if current is None else int(current) + 1
        await connection.execute(insert(table).values(**_row_values(self._scope_id, node)))

    async def _update_nodes(self, connection: AsyncConnection, nodes: Iterable[_Node]) -> None:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        for node in nodes:
            await connection.execute(
                update(table)
                .where(table.c.scope_id == self._scope_id, table.c.node_id == node.node_id)
                .values(
                    available=node.available,
                    last_sequence=node.last_sequence,
                    last_event_json=_event_json(node.last_event),
                )
            )

    def _database(self) -> Any:
        return self._runtime._provider.database

    def _topics(self) -> Any:
        return self._runtime._provider.repositories.topic_memories

    @property
    def _runtime(self) -> Any:
        return self._application._runtime


def _active_ids(nodes: Mapping[str, _Node]) -> list[str]:
    return sorted(node.node_id for node in nodes.values() if node.available)


def _same_binding(
    node: _Node,
    *,
    kind: str,
    policy: str | None,
    revision: int,
    ref: SourceRef | MemoryCitation | ArtifactRef,
    parents: tuple[str, ...],
) -> bool:
    return (
        node.kind == kind
        and node.policy == policy
        and node.lifecycle_revision == revision
        and node.ref == ref
        and node.parents == parents
    )


def _require_acyclic(nodes: Mapping[str, _Node], node_id: str, parents: tuple[str, ...]) -> None:
    if node_id in parents:
        raise SourceLifecycleError("cycle")
    if any(parent not in nodes for parent in parents):
        raise SourceLifecycleError("missing-parent")
    graph = {item.node_id: item.parents for item in nodes.values()}
    graph[node_id] = parents
    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(current: str) -> bool:
        if current in visiting:
            return True
        if current in visited:
            return False
        visiting.add(current)
        for parent in graph.get(current, ()):
            if walk(parent):
                return True
        visiting.remove(current)
        visited.add(current)
        return False

    if walk(node_id):
        raise SourceLifecycleError("cycle")


def _supported(nodes: Mapping[str, _Node], parents: tuple[str, ...], policy: str | None) -> bool:
    flags = [nodes[parent].available for parent in parents]
    if policy == "all":
        return all(flags)
    if policy == "any":
        return any(flags)
    raise SourceLifecycleError("policy")


def _recompute(nodes: dict[str, _Node]) -> None:
    children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    indegree = {node_id: 0 for node_id in nodes}
    for node in nodes.values():
        for parent in node.parents:
            if parent not in children:
                raise SourceLifecycleError("missing-parent")
            children[parent].append(node.node_id)
            indegree[node.node_id] += 1
    pending = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    ordered: list[str] = []
    while pending:
        current = pending.pop(0)
        ordered.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                pending.append(child)
        pending.sort()
    if len(ordered) != len(nodes):
        raise SourceLifecycleError("cycle")
    for node_id in ordered:
        node = nodes[node_id]
        if node.kind == "source":
            continue
        flags = [nodes[parent].available for parent in node.parents]
        if node.policy == "all":
            node.available = all(flags)
        elif node.policy == "any":
            node.available = any(flags)
        else:
            raise SourceLifecycleError("policy")


def _load_ref(kind: str, payload: Any) -> SourceRef | MemoryCitation | ArtifactRef:
    if kind == "source":
        return SourceRef.model_validate(payload)
    if kind == "memory":
        return MemoryCitation.model_validate(payload)
    if kind == "topic":
        return ArtifactRef.model_validate(payload)
    raise SourceLifecycleError("kind")


def _row_values(scope_id: str, node: _Node) -> dict[str, Any]:
    return {
        "scope_id": scope_id,
        "node_id": node.node_id,
        "ordinal": node.ordinal,
        "kind": node.kind,
        "policy": node.policy,
        "lifecycle_revision": node.lifecycle_revision,
        "ref_json": json.dumps(node.ref.model_dump(mode="json"), sort_keys=True, separators=(",", ":")),
        "parents_json": json.dumps(list(node.parents), separators=(",", ":")),
        "available": node.available,
        "last_sequence": node.last_sequence,
        "last_event_json": _event_json(node.last_event),
    }


def _event_json(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _node_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceLifecycleError("node-id")
    return value


def _policy(value: object) -> str:
    if value not in _POLICIES:
        raise SourceLifecycleError("policy")
    return str(value)


def _positive_int(value: object) -> int:
    if type(value) is not int or value < 1:
        raise SourceLifecycleError("revision")
    return value


def _parents(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SourceLifecycleError("parents")
    parents = tuple(value)
    if not parents or any(not isinstance(parent, str) or not parent.strip() for parent in parents):
        raise SourceLifecycleError("parents")
    if len(set(parents)) != len(parents):
        raise SourceLifecycleError("duplicate-parent")
    return parents
