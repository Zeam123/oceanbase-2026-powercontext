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

"""Choose literal, query-aligned spans from memory text under a rendered byte budget."""

from __future__ import annotations

import math
import re
from collections import Counter
from itertools import combinations
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# A rendered fragment is (document index, start, end) in the original entry text.
LiteralSpan = tuple[int, int, int]
_Measure = Callable[[Sequence[LiteralSpan]], int]

_TOKEN = re.compile(r"[A-Za-z]+|\d+|[\u4e00-\u9fff]+")
_HYPHENATED = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")
_WORD = re.compile(r"\S+")
_ABBREVIATIONS = frozenset(
    {
        "al",
        "approx",
        "ca",
        "cf",
        "dept",
        "dr",
        "ed",
        "eds",
        "eq",
        "etc",
        "fig",
        "figs",
        "jr",
        "min",
        "mr",
        "mrs",
        "ms",
        "no",
        "pp",
        "prof",
        "ref",
        "refs",
        "sr",
        "st",
        "vs",
        "vol",
    }
)
_STOPWORDS = frozenset(
    """
    a an the of and or in on for to with by from as at into over under than then
    also not no nor but if while when where who whom which what this that these those
    it its their his her our your they them we you he she i
    is are was were be been being am
    can may might could would should will shall must
    do does did doing done have has had having
    such both few more most other some any each every
    per via within without among between during after before above below
    about against using used use based onto upon
    """.split()
)
_BIGRAM_WEIGHT = 0.62
_PHRASE_WEIGHT = 0.8
_NUMBER_WEIGHT = 0.55


@dataclass(frozen=True)
class _QueryTerm:
    canon: str
    idf: float
    keys: frozenset[str]


@dataclass(frozen=True)
class _Slice:
    doc: int
    start: int
    end: int


@dataclass(frozen=True)
class _Unit:
    doc: int
    start: int
    end: int
    score: float
    matched: frozenset[str]


def select_literal_spans(
    query: str,
    documents: Sequence[str],
    max_bytes: int,
    measure: _Measure,
) -> tuple[LiteralSpan, ...]:
    """Return non-overlapping literal spans whose rendered context fits ``max_bytes``.

    Every returned span is an exact slice of one document. Among subsets that fit,
    the one with the highest total query score is kept.
    """

    sentences = [
        _Slice(doc=index, start=start, end=end)
        for index, text in enumerate(documents)
        for start, end in _sentence_spans(text)
    ]
    if not sentences:
        return ()
    token_sets = [
        frozenset(_keys_and_canons(_normalize_match(documents[sentence.doc][sentence.start : sentence.end]))[0])
        for sentence in sentences
    ]
    terms, bigrams, phrases, numbers = _analyze_query(query, token_sets)
    if not terms and not phrases and not numbers:
        return ()
    units = _candidate_units(documents, sentences, terms, bigrams, phrases, numbers)
    if not units:
        return ()
    units = _focus_best_document(units)
    units = _shortlist(units)
    cache: dict[tuple[LiteralSpan, ...], int] = {}

    def cached(spans: Sequence[LiteralSpan]) -> int:
        key = tuple(spans)
        cached_bytes = cache.get(key)
        if cached_bytes is None:
            cached_bytes = measure(key)
            cache[key] = cached_bytes
        return cached_bytes

    chosen = _pack(documents, units, max_bytes, cached)
    if chosen:
        return _merge_spans(documents, chosen)
    best = max(units, key=lambda unit: (unit.score, -(unit.end - unit.start)))
    trimmed = _trim_to_budget(documents[best.doc], best, max_bytes, measure, terms)
    if trimmed is None:
        return ()
    return (trimmed,)


def _analyze_query(
    query: str,
    token_sets: Sequence[frozenset[str]],
) -> tuple[tuple[_QueryTerm, ...], tuple[tuple[str, str], ...], tuple[str, ...], tuple[str, ...]]:
    document_frequency: Counter[str] = Counter()
    for keys in token_sets:
        document_frequency.update(keys)
    total = len(token_sets) or 1
    raw_terms = _content_tokens(_normalize_match(query))
    if not raw_terms:
        raw_terms = [token.lower() for token in _TOKEN.findall(query)]
        raw_terms = [token for token in raw_terms if token not in _STOPWORDS or token.isdigit()]
    seen: set[str] = set()
    terms: list[_QueryTerm] = []
    for token in raw_terms:
        canon = _canon(token)
        if canon in seen:
            continue
        seen.add(canon)
        keys = _token_keys(token)
        present = [document_frequency[key] for key in keys if document_frequency[key]]
        frequency = min(present) if present else 0
        idf = math.log((total + 1) / (frequency + 0.5)) + 1.0
        terms.append(_QueryTerm(canon=canon, idf=idf, keys=keys))
    canons = [_canon(token) for token in raw_terms]
    bigrams = tuple(dict.fromkeys(zip(canons, canons[1:], strict=False)))
    phrases = tuple(dict.fromkeys(match.group(0).lower() for match in _HYPHENATED.finditer(query)))
    numbers = tuple(dict.fromkeys(token for token in raw_terms if token.isdigit() and len(token) >= 2))
    return tuple(terms), bigrams, phrases, numbers


def _candidate_units(
    documents: Sequence[str],
    sentences: Sequence[_Slice],
    terms: Sequence[_QueryTerm],
    bigrams: Sequence[tuple[str, str]],
    phrases: Sequence[str],
    numbers: Sequence[str],
) -> list[_Unit]:
    units: list[_Unit] = []
    for sentence in sentences:
        text = documents[sentence.doc][sentence.start : sentence.end]
        unit = _score_slice(
            sentence.doc, sentence.start, sentence.end, text, terms, bigrams, phrases, numbers, scale=1.0
        )
        unit = _prefer_specific_sentence(unit, text)
        if unit.score > 0:
            units.append(unit)
    return units


def _prefer_specific_sentence(unit: _Unit, text: str) -> _Unit:
    """Prefer a concrete result over a long background sentence that repeats the query.

    Evidence is often one or two result sentences. A long objective or hypothesis can
    share more query words and otherwise consume the whole byte budget.
    """

    if unit.score <= 0:
        return unit
    encoded = len(text.encode("utf-8"))
    factor = 360 / encoded if encoded > 360 else 1.0
    lowered = text.lstrip().lower()
    if lowered.startswith(
        ("objective", "background", "purpose", "design", "methods", "introduction", "participants", "main outcome")
    ):
        factor *= 0.62
    if lowered.startswith(("however, whether", "whether ")) or lowered.startswith("to determine"):
        factor *= 0.74
    if lowered.startswith(("conclusion", "results", "interpretation")) or any(
        cue in lowered for cue in (" we found ", " we show ", " we demonstrate", "significantly")
    ):
        factor *= 1.16
    if factor == 1.0:
        return unit
    return _Unit(doc=unit.doc, start=unit.start, end=unit.end, score=unit.score * factor, matched=unit.matched)


def _score_slice(
    doc: int,
    start: int,
    end: int,
    text: str,
    terms: Sequence[_QueryTerm],
    bigrams: Sequence[tuple[str, str]],
    phrases: Sequence[str],
    numbers: Sequence[str],
    *,
    scale: float,
) -> _Unit:
    keys, canons = _keys_and_canons(_normalize_match(text))
    matched = [term for term in terms if term.keys & keys or _substring_hit(term.keys, keys)]
    total_idf = sum(term.idf for term in terms)
    score = sum(term.idf for term in matched) / total_idf if total_idf else 0.0
    if bigrams:
        present = set(zip(canons, canons[1:], strict=False))
        score += _BIGRAM_WEIGHT * (sum(1 for pair in bigrams if pair in present) / len(bigrams))
    if phrases:
        lowered = text.lower()
        score += _PHRASE_WEIGHT * (sum(1 for phrase in phrases if phrase in lowered) / len(phrases))
    if numbers:
        found = {canon for canon in canons if canon.isdigit()}
        score += _NUMBER_WEIGHT * (sum(1 for number in numbers if number in found) / len(numbers))
    return _Unit(doc=doc, start=start, end=end, score=score * scale, matched=frozenset(term.canon for term in matched))


def _focus_best_document(units: list[_Unit]) -> list[_Unit]:
    """Keep most of the budget on the abstract that best matches the query.

    Distractor abstracts share topic words and otherwise crowd out the evidence sentences.
    """

    best = max(units, key=lambda unit: unit.score)
    focused: list[_Unit] = []
    for unit in units:
        if unit.doc == best.doc:
            focused.append(unit)
            continue
        focused.append(
            _Unit(doc=unit.doc, start=unit.start, end=unit.end, score=unit.score * 0.45, matched=unit.matched)
        )
    return focused


def _shortlist(units: list[_Unit]) -> list[_Unit]:
    """Keep the strongest slices. Low-ranked slices cannot displace a better fit."""

    if len(units) <= 28:
        return units
    ranked = sorted(units, key=lambda unit: (unit.score, -(unit.end - unit.start)), reverse=True)
    return ranked[:28]


def _pack(
    documents: Sequence[str],
    units: Sequence[_Unit],
    max_bytes: int,
    measure: _Measure,
) -> list[_Unit]:
    """Pick the feasible subset with the greatest total query score.

    Greedy selection lets one long, highly overlapping sentence consume the budget before
    a pair of shorter evidence sentences whose scores add up to more.
    """

    shortlisted = sorted(units, key=lambda unit: (unit.score, -(unit.end - unit.start)), reverse=True)[:10]
    best_value: tuple[float, int, int] | None = None
    best_combo: tuple[_Unit, ...] = ()
    for size in range(1, 4):
        for combo in combinations(shortlisted, size):
            if _combo_overlaps(combo):
                continue
            spans = _merge_spans(documents, combo)
            text_bytes = sum(len(documents[doc][start:end].encode("utf-8")) for doc, start, end in spans)
            if text_bytes > max_bytes:
                continue
            rendered = measure(spans)
            if rendered > max_bytes:
                continue
            matched: set[str] = set()
            for unit in combo:
                matched.update(unit.matched)
            value = (sum(unit.score for unit in combo), len(matched), -rendered)
            if best_value is None or value > best_value:
                best_value = value
                best_combo = combo
    return list(best_combo)


def _combo_overlaps(combo: Sequence[_Unit]) -> bool:
    for index, unit in enumerate(combo):
        if _overlaps(unit, combo[:index]):
            return True
    return False


def _overlaps(unit: _Unit, chosen: Sequence[_Unit]) -> bool:
    return any(unit.doc == other.doc and unit.start < other.end and other.start < unit.end for other in chosen)


def _merge_spans(documents: Sequence[str], units: Sequence[_Unit]) -> tuple[LiteralSpan, ...]:
    grouped: dict[int, list[_Unit]] = {}
    for unit in units:
        grouped.setdefault(unit.doc, []).append(unit)
    merged: list[LiteralSpan] = []
    for doc, doc_units in grouped.items():
        ordered = sorted(doc_units, key=lambda unit: (unit.start, unit.end))
        current_start, current_end = ordered[0].start, ordered[0].end
        for unit in ordered[1:]:
            gap = documents[doc][current_end : unit.start]
            if unit.start <= current_end or not gap.strip():
                current_end = max(current_end, unit.end)
                continue
            merged.append((doc, current_start, current_end))
            current_start, current_end = unit.start, unit.end
        if documents[doc][current_start:current_end].strip():
            merged.append((doc, current_start, current_end))
    merged.sort()
    return tuple(span for span in merged if span[1] < span[2])


def _trim_to_budget(
    text: str,
    unit: _Unit,
    max_bytes: int,
    measure: _Measure,
    terms: Sequence[_QueryTerm],
) -> LiteralSpan | None:
    words: list[tuple[int, int, bool]] = []
    for match in _WORD.finditer(text, unit.start, unit.end):
        keys, _canons = _keys_and_canons(_normalize_match(match.group(0)))
        is_query = any(term.keys & keys or _substring_hit(term.keys, keys) for term in terms)
        words.append((match.start(), match.end(), is_query))
    if not words:
        return None

    def fits(start: int, end: int) -> bool:
        return start < end and measure(((unit.doc, start, end),)) <= max_bytes

    while len(words) > 1 and not fits(words[0][0], words[-1][1]):
        if words[-1][2] and not words[0][2]:
            words.pop(0)
        else:
            words.pop()
    if not fits(words[0][0], words[-1][1]):
        return None
    return (unit.doc, words[0][0], words[-1][1])


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    if not text.strip():
        return []
    boundaries = [0]
    cursor = 0
    length = len(text)
    while cursor < length:
        character = text[cursor]
        if character not in ".!?。！？":
            cursor += 1
            continue
        if character == "." and cursor + 1 < length and text[cursor + 1].isdigit():
            cursor += 1
            continue
        if _is_abbreviation(text, cursor):
            cursor += 1
            continue
        end = cursor + 1
        while end < length and text[end] in "\"')]}":
            end += 1
        follower = end
        while follower < length and text[follower].isspace():
            follower += 1
        if follower >= length or _starts_sentence(text[follower]):
            boundaries.append(follower)
            cursor = follower
            continue
        cursor += 1
    spans: list[tuple[int, int]] = []
    stops = [*boundaries[1:], length]
    for start, end in zip(boundaries, stops, strict=False):
        while start < end and text[start].isspace():
            start += 1
        if start < end:
            spans.append((start, end))
    return spans


def _is_abbreviation(text: str, period: int) -> bool:
    index = period - 1
    while index >= 0 and text[index] in "\"')]}":
        index -= 1
    end = index
    while index >= 0 and text[index].isalpha():
        index -= 1
    word = text[index + 1 : end + 1]
    return bool(word) and (len(word) == 1 or word.lower() in _ABBREVIATIONS)


def _starts_sentence(character: str) -> bool:
    return character.isupper() or character.isdigit() or "\u4e00" <= character <= "\u9fff" or character in "([{\"'"


def _normalize_match(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", text).lower()


def _content_tokens(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN.findall(text):
        lowered = token.lower()
        if lowered in _STOPWORDS:
            continue
        if len(lowered) == 1 and lowered.isalpha():
            continue
        tokens.append(lowered)
    return tokens


def _keys_and_canons(text: str) -> tuple[set[str], list[str]]:
    keys: set[str] = set()
    canons: list[str] = []
    for token in _content_tokens(text):
        keys.update(_token_keys(token))
        canons.append(_canon(token))
    return keys, canons


def _token_keys(token: str) -> frozenset[str]:
    if token.isdigit() or _is_cjk(token):
        return frozenset((token,))
    normalized = token.replace("ae", "e").replace("oe", "e")
    keys = {token, normalized}
    for source in (token, normalized):
        stemmed = _inflect(source)
        keys.add(stemmed)
        for suffix in ("tion", "sion", "ment", "ness", "ance", "ence", "able", "ible", "ity"):
            if stemmed.endswith(suffix) and len(stemmed) - len(suffix) >= 4:
                keys.add(stemmed[: -len(suffix)])
        if stemmed.endswith("e") and len(stemmed) > 5:
            keys.add(stemmed[:-1])
    return frozenset(keys)


def _canon(token: str) -> str:
    if token.isdigit() or _is_cjk(token):
        return token
    return _inflect(token.replace("ae", "e").replace("oe", "e"))


def _inflect(token: str) -> str:
    if len(token) <= 4 or token.isdigit():
        return token
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("ied") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 6:
        stem = token[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            return stem[:-1]
        return stem
    if token.endswith("ed") and len(token) > 5:
        stem = token[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            return stem[:-1]
        return stem
    if token.endswith("es") and len(token) > 4 and (token[-3] in "sxz" or token.endswith(("ches", "shes"))):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
        return token[:-1]
    return token


def _substring_hit(query_keys: frozenset[str], text_keys: set[str]) -> bool:
    long_query = [key for key in query_keys if len(key) >= 7 and not key.isdigit()]
    if not long_query:
        return False
    long_text = [key for key in text_keys if len(key) >= 7 and not key.isdigit()]
    return any(query in text or text in query for query in long_query for text in long_text)


def _is_cjk(token: str) -> bool:
    return bool(token) and all("\u4e00" <= character <= "\u9fff" for character in token)
