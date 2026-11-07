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

"""Pi session-branch Sources that keep original events available after restart."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
_IGNORED_BLOCKS = frozenset({"thinking", "image", "redacted_thinking"})


class PiSessionInput(BaseModel):
    """One Pi v3 JSONL session and the leaf that selects its history branch."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected branch, including its original record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """Captured Pi event whose identity is the session id plus the event id."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class _ParsedSession(NamedTuple):
    session_id: str
    records: dict[str, dict[str, JsonValue]]
    events: tuple[PiEventInput, ...]


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse one session and return the selected branch from root to leaf."""

    return _parse_session(value.content, value.leaf_id).events


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build a stable Source id from a session id and an event id."""

    readable = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(readable) <= MAX_SOURCE_ID_LENGTH:
        return readable
    digest = hashlib.sha256(session_id.encode() + b"\0" + event_id.encode()).hexdigest()
    return f"pih:{digest}"


class PiEventSourceAdapter:
    """Resolve one captured Pi event and read that same event back."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            definition_version="1",
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Project one Pi event as canonical text evidence without shortening it."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PiEventSourceAdapter(),
    version="1",
    projections=(PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Import new visible events from one Pi file without advancing past a bad read."""

    name = PI_SESSION_CONNECTOR_NAME
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        try:
            content = self._path.read_bytes()
        except OSError as error:
            raise ValueError("Pi session file cannot be read") from error
        parsed = _parse_session(content, self._leaf_id)
        saved_session, imported = _imported_events(session.checkpoint)
        if saved_session is not None and saved_session != parsed.session_id:
            raise ValueError("imported Pi session was replaced")
        for event_id, digest in imported.items():
            record = parsed.records.get(event_id)
            if record is None or _record_digest(record) != digest:
                raise ValueError("imported Pi event was rewritten")

        updated = dict(imported)
        for event in parsed.events:
            if event.event_id in updated:
                continue
            await session.submit(
                pi_event_source_id(event.session_id, event.event_id),
                PI_EVENT_SOURCE_NAME,
                event,
            )
            updated[event.event_id] = _record_digest(parsed.records[event.event_id])
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"session_id": parsed.session_id, "events": updated},
        )


def _parse_session(content: bytes, leaf_id: str | None) -> _ParsedSession:
    rows = _load_rows(content)
    session_id, records, order = _build_graph(rows)
    if leaf_id is None:
        selected = order[-1]
    elif leaf_id not in records:
        raise ValueError("Pi session leaf does not exist")
    else:
        selected = leaf_id
    visible = {event_id: _render_event(session_id, records[event_id]) for event_id in order}
    events = tuple(
        event for event_id in _ancestor_ids(records, selected) if (event := visible[event_id]) is not None
    )
    return _ParsedSession(session_id, records, events)


def _load_rows(content: bytes) -> list[dict[str, JsonValue]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session must be UTF-8") from error
    rows: list[dict[str, JsonValue]] = []
    for line in text.splitlines():
        if not line.strip():
            raise ValueError("Pi session line must be JSON")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("Pi session line must be JSON") from error
        if not isinstance(value, dict):
            raise ValueError("Pi session line must be a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError("Pi session is empty")
    return rows


def _build_graph(
    rows: list[dict[str, JsonValue]],
) -> tuple[str, dict[str, dict[str, JsonValue]], list[str]]:
    header = rows[0]
    version = header.get("version")
    session_id = header.get("id")
    if header.get("type") != "session" or type(version) is not int or version != 3:
        raise ValueError("Pi session header must be type session version 3")
    if not isinstance(session_id, str) or session_id == "":
        raise ValueError("Pi session id must be a non-empty string")

    records: dict[str, dict[str, JsonValue]] = {}
    order: list[str] = []
    for record in rows[1:]:
        event_id = record.get("id")
        event_type = record.get("type")
        if not isinstance(event_id, str) or event_id == "":
            raise ValueError("Pi event id must be a non-empty string")
        if event_id in records:
            raise ValueError("Pi event id is duplicated")
        if "parentId" not in record:
            raise ValueError("Pi event is missing parentId")
        parent_id = record.get("parentId")
        if parent_id is not None and (not isinstance(parent_id, str) or parent_id not in records):
            raise ValueError("Pi event parent does not exist")
        if not isinstance(event_type, str) or event_type == "":
            raise ValueError("Pi event type must be a non-empty string")
        records[event_id] = record
        order.append(event_id)
    if not order:
        raise ValueError("Pi session has no events")
    return session_id, records, order


def _ancestor_ids(records: dict[str, dict[str, JsonValue]], leaf_id: str) -> tuple[str, ...]:
    chain: list[str] = []
    seen: set[str] = set()
    current: str | None = leaf_id
    while current is not None:
        if current in seen:
            raise ValueError("Pi session contains a cycle")
        seen.add(current)
        chain.append(current)
        parent_id = records[current].get("parentId")
        current = parent_id if isinstance(parent_id, str) else None
    chain.reverse()
    return tuple(chain)


def _render_event(session_id: str, record: dict[str, JsonValue]) -> PiEventInput | None:
    event_type = record["type"]
    if not isinstance(event_type, str):
        raise ValueError("Pi event type must be a string")
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            raise ValueError("Pi message entry requires an object")
        body = _message_body(message)
        role = message.get("role")
    elif event_type in {"compaction", "branch_summary"}:
        if "summary" not in record or record.get("summary") is None:
            return None
        summary = record.get("summary")
        if not isinstance(summary, str):
            raise ValueError("Pi summary must be a string")
        body = summary if summary else None
        role = event_type
    elif event_type == "custom_message":
        body = _content_body(record.get("content"))
        role = "custom"
    else:
        return None
    if body is None:
        return None
    if not isinstance(role, str) or role == "":
        raise ValueError("Pi event role must be a non-empty string")
    parent_id = record.get("parentId")
    event_id = record.get("id")
    if not isinstance(event_id, str) or (parent_id is not None and not isinstance(parent_id, str)):
        raise ValueError("Pi event identity is invalid")
    return PiEventInput(
        session_id=session_id,
        event_id=event_id,
        parent_id=parent_id,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _message_body(message: dict[str, JsonValue]) -> str | None:
    role = message.get("role")
    if role == "toolResult":
        text = _content_body(message.get("content")) or ""
        tool_name = message.get("toolName", "")
        tool_call_id = message.get("toolCallId", "")
        if not isinstance(tool_name, str) or not isinstance(tool_call_id, str):
            raise ValueError("Pi tool result name and call id must be strings")
        if text:
            return f"[tool_result {tool_name} {tool_call_id}] {text}"
        return f"[tool_result {tool_name} {tool_call_id}]"
    if role == "bashExecution":
        command = message.get("command", "")
        output = message.get("output", "")
        if not isinstance(command, str) or not isinstance(output, str):
            raise ValueError("Pi bash execution command and output must be strings")
        return f"$ {command}\n{output}"
    return _content_body(message.get("content"))


def _content_body(content: JsonValue | None) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content if content else None
    if not isinstance(content, list):
        raise ValueError("Pi event content must be a string or a list of blocks")
    parts: list[str] = []
    for block in content:
        part = _block_text(block)
        if part is not None:
            parts.append(part)
    if not parts:
        return None
    rendered = "\n".join(parts)
    return rendered if rendered else None


def _block_text(block: JsonValue) -> str | None:
    if not isinstance(block, dict) or not isinstance(block.get("type"), str):
        raise ValueError("Pi content block must identify its type")
    kind = block["type"]
    if kind == "text":
        text = block.get("text", "")
        if not isinstance(text, str):
            raise ValueError("Pi text block must contain text")
        return text
    if kind == "toolCall":
        return _tool_call(block)
    if kind in _IGNORED_BLOCKS:
        return None
    return None


def _tool_call(block: dict[str, JsonValue]) -> str:
    name = block.get("name", "")
    call_id = block.get("id", "")
    if not isinstance(name, str) or not isinstance(call_id, str):
        raise ValueError("Pi tool call name and id must be strings")
    arguments = block.get("arguments", {})
    try:
        encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("Pi tool call arguments must be JSON") from error
    return f"[tool_call {name} {call_id}] {encoded}"


def _record_digest(record: dict[str, JsonValue]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _imported_events(checkpoint: JsonValue | None) -> tuple[str | None, dict[str, str]]:
    if checkpoint is None:
        return None, {}
    if not isinstance(checkpoint, dict):
        raise ValueError("Pi checkpoint must be an object")
    session_id = checkpoint.get("session_id")
    events = checkpoint.get("events")
    if not isinstance(session_id, str) or session_id == "" or not isinstance(events, dict):
        raise ValueError("Pi checkpoint is missing imported events")
    imported: dict[str, str] = {}
    for event_id, digest in events.items():
        if not isinstance(event_id, str) or event_id == "" or not isinstance(digest, str) or digest == "":
            raise ValueError("Pi checkpoint event is invalid")
        imported[event_id] = digest
    return session_id, imported
