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

"""Row-level CSV Sources that keep original fields available for citation lookup."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.adapters import SourceAdapter
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document addressed by a caller-stable document id."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One validated CSV row, with column order and original field values preserved."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """Captured row evidence whose identity is the document id plus primary key."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse one offline CSV document into row inputs, or fail without partial rows."""

    _validate_document_id(value.document_id)
    if value.primary_key == "":
        raise ValueError("primary_key must be non-empty")

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be strict UTF-8") from error

    reader = csv.reader(
        io.StringIO(text, newline=""),
        delimiter=",",
        quotechar='"',
        doublequote=True,
        escapechar=None,
        strict=True,
        skipinitialspace=False,
    )
    try:
        records = list(reader)
    except csv.Error as error:
        raise ValueError("CSV content is malformed") from error

    if not records or not records[0]:
        raise ValueError("CSV header must not be empty")

    columns = tuple(records[0])
    _validate_columns(columns)
    try:
        primary_key_index = columns.index(value.primary_key)
    except ValueError as error:
        raise ValueError("primary_key must match a header field exactly") from error

    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    width = len(columns)
    for record in records[1:]:
        if not record:
            continue
        if len(record) != width:
            raise ValueError("CSV record width must match the header")
        primary_key_value = record[primary_key_index]
        if primary_key_value == "" or primary_key_value.strip() == "":
            raise ValueError("primary key value must not be blank")
        if primary_key_value in seen_keys:
            raise ValueError("primary key values must be unique within the document")
        seen_keys.add(primary_key_value)
        csv_row_source_id(value.document_id, primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=value.document_id,
                primary_key=value.primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(rows)


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the stable row identity, percent-encoding both components."""

    _validate_document_id(document_id)
    if not isinstance(primary_key_value, str) or primary_key_value == "" or primary_key_value.strip() == "":
        raise ValueError("primary key value must not be blank")
    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


class CSVRowSourceAdapter(SourceAdapter[CSVRowInput, CSVRowSource, CSVRowInput]):
    """Resolve one captured row and read its original fields back unchanged."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        _validate_row_input(value)
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            definition_version="1",
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=tuple(value.columns),
            values=tuple(value.values),
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=tuple(source.columns),
            values=tuple(source.values),
        )


class CSVRowTextEvidenceProjection:
    """Project one row as canonical text evidence without changing field values."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = {column: value for column, value in zip(source.columns, source.values, strict=True)}
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.name,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSVRowSourceAdapter(),
    version="1",
    projections=(CSVRowTextEvidenceProjection(),),
)


def _validate_document_id(document_id: str) -> None:
    if not isinstance(document_id, str) or document_id == "" or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty string without surrounding whitespace")


def _validate_columns(columns: tuple[str, ...]) -> None:
    seen: set[str] = set()
    for column in columns:
        if column == "" or column.strip() == "":
            raise ValueError("CSV header fields must be non-empty and not whitespace")
        if column in seen:
            raise ValueError("CSV header fields must be unique")
        seen.add(column)


def _validate_row_input(value: CSVRowInput) -> None:
    _validate_document_id(value.document_id)
    if value.primary_key == "":
        raise ValueError("primary_key must be non-empty")
    columns = tuple(value.columns)
    values = tuple(value.values)
    _validate_columns(columns)
    if len(values) != len(columns):
        raise ValueError("CSV row width must match the header")
    try:
        primary_key_index = columns.index(value.primary_key)
    except ValueError as error:
        raise ValueError("primary_key must match a header field exactly") from error
    if values[primary_key_index] != value.primary_key_value:
        raise ValueError("primary_key_value must match the primary key field")
    if value.primary_key_value == "" or value.primary_key_value.strip() == "":
        raise ValueError("primary key value must not be blank")
