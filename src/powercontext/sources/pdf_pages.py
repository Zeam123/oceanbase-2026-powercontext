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

"""Page-level PDF Sources and offline retrieval over scoped Memory."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from io import BytesIO
from urllib.parse import quote

from pydantic import BaseModel
from pypdf import PdfReader

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PDF_PAGE_SOURCE_NAME = "pdf-page"
_TOKEN = re.compile(r"[a-z]+|\d+")
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "from",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "what",
        "which",
        "who",
        "how",
        "why",
        "when",
        "where",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "that",
        "this",
        "these",
        "those",
        "if",
        "then",
        "not",
        "no",
        "any",
        "has",
        "have",
        "had",
        "it",
        "its",
        "their",
        "our",
        "we",
        "you",
        "than",
        "into",
        "over",
        "about",
        "after",
        "before",
        "such",
        "only",
        "also",
        "using",
        "use",
        "used",
        "within",
        "shown",
        "show",
        "give",
        "given",
        "answer",
        "question",
        "respond",
        "response",
        "calculate",
        "define",
        "defined",
        "assuming",
        "assume",
        "perspective",
        "details",
        "detail",
        "relying",
        "rely",
        "state",
        "outlined",
        "explicitly",
        "directly",
        "quantity",
        "reported",
        "report",
        "materially",
        "important",
        "ongoing",
        "unadjusted",
        "analyst",
        "investment",
        "usd",
        "millions",
        "million",
        "fy",
    }
)
_SYNONYMS = {
    "legal": ("litigation", "proceedings", "lawsuit"),
    "battles": ("litigation", "proceedings"),
    "geographies": ("geographic", "geographical", "geography"),
    "geography": ("geographic", "geographical"),
    "capex": ("expenditures", "expenditure"),
    "ebitda": ("depreciation", "amortization"),
    "income": ("profit", "earnings", "operations"),
    "profit": ("income", "earnings"),
    "revenue": ("revenues", "sales"),
    "revenues": ("revenue", "sales"),
    "sales": ("revenue", "revenues"),
    "margin": ("profit",),
    "drove": ("driven", "increase"),
    "customers": ("customer", "revenues"),
    "customer": ("customers",),
    "concentration": ("accounted",),
    "cogs": ("cost", "sales"),
    "cyclicality": ("cyclical", "cycle"),
    "cyclical": ("cyclicality",),
}
# Question cues, short Memory queries, statement families, and phrases that mark a narrative page.
_INTENTS = (
    (("quick ratio", "current ratio", "liquidity profile"), ("current assets", "cash equivalents"), ("balance",), ()),
    (("working capital",), ("current assets", "current liabilities"), ("balance",), ()),
    (
        ("capital-intensive", "capital intensive", "capital expenditure", "capital expenditures", "capex"),
        ("capital expenditures", "cash flows", "net sales"),
        ("cash", "income", "balance"),
        (),
    ),
    (("days payable", "accounts payable", "cogs"), ("accounts payable", "cost of sales"), ("balance", "income"), ()),
    (("inventory turnover",), ("inventories", "cost of sales"), ("balance", "income"), ()),
    (("gross margin", "gross profit"), ("gross profit", "cost of sales"), ("income",), ()),
    (("operating margin",), ("operating income",), ("income",), ("operating income",)),
    (("ebitda",), ("depreciation", "operating income"), ("income", "cash"), ()),
    (("effective tax", "tax rate"), ("income tax",), ("income",), ("income tax",)),
    (
        ("p&l", "income statement", "statement of income", "statements of income", "statement of operations"),
        ("net sales", "total revenues", "operating income"),
        ("income",),
        (),
    ),
    (("cash flow",), ("cash flows", "operating activities"), ("cash",), ()),
    (("balance sheet", "financial position"), ("current assets",), ("balance",), ()),
    (("segment",), ("business segment", "organic sales"), (), ("business segment", "organic sales", "segment data")),
    (
        ("customer concentration", "primary customers", "customers", "customer"),
        ("customer accounted", "revenues earned", "government contracts"),
        (),
        ("accounted for", "portion of our revenue", "revenues were earned", "government contracts"),
    ),
    (
        ("products and services", "product categories", "service categories", "major products"),
        ("products services", "business segment"),
        (),
        ("primarily offering", "item 1. business", "item 1.business", "business segment", "segment data"),
    ),
    (
        ("drove", "revenue change", "change in revenue"),
        ("net revenue", "driven"),
        (),
        ("driven by", "results of operations"),
    ),
    (("geograph",), ("geographic operates", "geographical"), (), ("geographic", "geographical", "operates")),
    (("restructuring",), ("restructuring impairment",), (), ("restructuring and impairment", "restructuring charges")),
    (("legal", "litigation"), ("legal proceedings",), (), ("legal proceedings",)),
    (("acquisition", "acquisitions"), ("acquisition acquired",), (), ("acquisition",)),
    (("dividend",), ("dividends",), (), ("dividends",)),
)


class PDFDocumentInput(BaseModel):
    """One complete PDF addressed by a caller-stable document id."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical PDF page, including an empty page with no extractable text."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Captured page evidence whose identity includes the PDF revision."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page, or fail without returning a partial document."""

    _validate_document_id(value.document_id)
    if not isinstance(value.content, bytes) or not value.content:
        raise ValueError("PDF content must be a non-empty document")
    try:
        texts = _page_texts(value.content)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("PDF cannot be parsed") from error
    digest = hashlib.sha256(value.content).hexdigest()
    return tuple(
        PDFPageInput(
            document_id=value.document_id,
            document_sha256=digest,
            page_number=page_number,
            text=text,
        )
        for page_number, text in enumerate(texts, start=1)
    )


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build a stable Source id from a document, its exact bytes, and a page number."""

    readable = f"pdf:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(readable) <= MAX_SOURCE_ID_LENGTH:
        return readable
    digest = hashlib.sha256(f"{document_id}\0{document_sha256}\0{page_number}".encode()).hexdigest()
    return f"pdfh:{digest}"


class PDFPageSourceAdapter:
    """Resolve one captured PDF page and read that same page back."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            definition_version="1",
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Project one PDF page as canonical text evidence anchored to that page."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDFPageSourceAdapter(),
    version="1",
    projections=(PDFPageTextEvidenceProjection(),),
)


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple:
    """Return at most ``limit`` citations for distinct pages related to ``question``."""

    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime import GetMemoryEntryRequest, SearchMemoryRequest

    if limit < 1 or not question or not question.strip():
        return ()
    hits = await _collect_hits(memory, question, SearchMemoryRequest)
    if not hits:
        return ()
    ranked: list[tuple[float, str, MemoryCitation]] = []
    seen_entries: set[tuple[str, str]] = set()
    prepared: list[tuple[str, MemoryCitation, str]] = []
    for hit in hits:
        citation = MemoryCitation(
            memory_ref=hit.memory_ref,
            entry_id=hit.entry_id,
            entry_version_id=hit.entry_version_id,
        )
        identity = (hit.entry_id, hit.entry_version_id)
        if identity in seen_entries:
            continue
        seen_entries.add(identity)
        try:
            record = await memory.get(GetMemoryEntryRequest(citation=citation))
        except Exception:
            continue
        source_ids = tuple(source.source_id for source in record.entry.sources if getattr(source, "source_id", ""))
        if not source_ids:
            continue
        prepared.append((hit.text or "", citation, source_ids[0]))
    if not prepared:
        return ()
    bags = [Counter(_tokens(text)) for text, _, _ in prepared]
    document_frequency: Counter[str] = Counter()
    for bag in bags:
        document_frequency.update(bag.keys())
    total = len(bags)
    terms = _expanded_terms(question)
    lowered_question = question.lower()
    requested_statements = _statement_kinds(lowered_question)
    for index, ((text, citation, source_id), bag) in enumerate(zip(prepared, bags, strict=True)):
        matched = []
        for term in terms:
            count = bag[term]
            if not count or document_frequency[term] > max(2, total * 0.5):
                continue
            idf = math.log((total + 1) / (document_frequency[term] + 0.5))
            matched.append(idf * (1 + math.log(count)))
        lexical = 0.0 if not matched else max(matched) * 6 + sum(matched)
        score = lexical + _page_bonus(lowered_question, text, requested_statements) - index * 0.001
        ranked.append((score, source_id, citation))
    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen = []
    seen_pages: set[str] = set()
    for _, source_id, citation in ranked:
        if source_id in seen_pages:
            continue
        seen_pages.add(source_id)
        chosen.append(citation)
        if len(chosen) >= limit:
            break
    return tuple(chosen)


def _validate_document_id(document_id: str) -> None:
    if not isinstance(document_id, str) or document_id.strip() == "":
        raise ValueError("document_id must not be empty or whitespace")


def _page_texts(content: bytes) -> tuple[str, ...]:
    reader = PdfReader(BytesIO(content), strict=False)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF cannot be read")
    texts: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text()
        texts.append(extracted if isinstance(extracted, str) else "")
    if not texts:
        raise ValueError("PDF has no pages")
    return tuple(texts)


async def _collect_hits(memory, question: str, request_type):
    found = []
    seen: set[tuple[str, str]] = set()
    for query, pool in _search_queries(question):
        try:
            page = await memory.search(request_type(query=query, limit=pool, mode="fts"))
        except Exception:
            continue
        for hit in page.hits:
            identity = (hit.entry_id, hit.entry_version_id)
            if identity in seen:
                continue
            seen.add(identity)
            found.append(hit)
    return found


def _search_queries(question: str) -> tuple[tuple[str, int], ...]:
    queries = [(question, 60)]
    ranked = sorted(_content_terms(question), key=len, reverse=True)
    for term in ranked[:8]:
        companion = _companion(term)
        queries.append((f"{term} {companion}" if companion != term else term, 30))
        queries.extend((f"{term} {synonym}", 30) for synonym in _SYNONYMS.get(term, ())[:2])
    lowered = question.lower()
    for _needles, extra, _kinds, _markers in _matching_intents(lowered):
        queries.extend((query, 30) for query in extra)
    if "m&a" in lowered or "m & a" in lowered:
        queries.extend((("organic sales", 30), ("business segment", 30), ("acquisitions divestitures", 30)))
    kinds = _statement_kinds(lowered)
    if "income" in kinds:
        queries.extend((("statements operations", 30), ("statements income", 30), ("statements earnings", 30)))
    if "balance" in kinds:
        queries.append(("balance sheets", 30))
    if "cash" in kinds:
        queries.append(("cash flows", 30))
    return tuple(dict.fromkeys(queries))


def _content_terms(text: str) -> list[str]:
    terms = []
    for token in _tokens(text):
        if token in _STOP or len(token) <= 2:
            continue
        terms.append(token)
        if token.startswith("fy") and token[2:].isdigit():
            terms.append(token[2:])
    return list(dict.fromkeys(terms))


def _expanded_terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in _content_terms(text):
        terms.extend(_variants(token))
        terms.extend(_SYNONYMS.get(token, ()))
    return list(dict.fromkeys(term for term in terms if term not in _STOP and len(term) > 2))


def _companion(term: str) -> str:
    forms = _variants(term) - {term}
    if not forms:
        return term
    return max(forms, key=len)


def _variants(word: str) -> set[str]:
    forms = {word}
    if word.endswith("ies") and len(word) > 5:
        forms.add(word[:-3] + "y")
    if word.endswith("es") and len(word) > 5:
        forms.add(word[:-2])
    if word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        forms.add(word[:-1])
    for form in list(forms):
        if form.endswith("e") and len(form) > 4:
            forms.add(form[:-1] + "ing")
        if form.endswith("y") and len(form) > 4:
            forms.update((form[:-1] + "ic", form[:-1] + "ical", form[:-1] + "ies"))
        if not form.endswith("s"):
            forms.add(form + "s")
    return {form for form in forms if len(form) > 2}


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _matching_intents(question: str):
    return tuple(intent for intent in _INTENTS if any(needle in question for needle in intent[0]))


def _statement_kinds(question: str) -> frozenset[str]:
    kinds = {kind for _needles, _queries, intent_kinds, _markers in _matching_intents(question) for kind in intent_kinds}
    return frozenset(kinds)


def _page_bonus(question: str, text: str, statements: frozenset[str]) -> float:
    normalized = " ".join(text.lower().split())
    bonus = 48.0 if _statement_kind(normalized) in statements else 0.0
    for _needles, _queries, _kinds, markers in _matching_intents(question):
        if any(marker in normalized for marker in markers):
            bonus += 24
    if "m&a" in question or "m & a" in question:
        if "organic sales" in normalized or "business segment" in normalized:
            bonus += 24
    return bonus


def _statement_kind(text: str) -> str | None:
    if sum(token.isdigit() for token in _TOKEN.findall(text)) < 12:
        return None
    head = text[:700]
    if any(phrase in head for phrase in ("statement of cash flow", "statements of cash flow")):
        return "cash"
    if "balance sheet" in head or "statement of financial position" in head or "statements of financial position" in head:
        return "balance"
    if any(
        phrase in head
        for phrase in (
            "statement of income",
            "statements of income",
            "statement of operations",
            "statements of operations",
            "statement of earnings",
            "statements of earnings",
            "statement of profit",
            "statements of profit",
        )
    ):
        return "income"
    return None
