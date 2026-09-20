"""Small, portable lexical RAG: Chinese bigrams and SQLite FTS5/BM25.

No embedding service or model-specific vector dimensions are needed. The same
terms support a BM25 fallback on SQLite builds which do not include FTS5.
"""

import math
import re
from collections import Counter

SEARCH_VERSION = 1
MAX_QUERY_TERMS = 96
MAX_INDEX_CHARS = 8000
MAX_EXCERPT_CHARS = 500
_COMMON = {
    "哈哈",
    "呵呵",
    "嘿嘿",
    "这个",
    "那个",
    "什么",
    "怎么",
    "你们",
    "我们",
    "他们",
    "就是",
    "不是",
    "一个",
    "一下",
    "现在",
    "然后",
    "但是",
    "已经",
    "还是",
    "因为",
    "所以",
    "没有",
    "真的",
    "可以",
    "知道",
    "感觉",
    "好像",
    "the",
    "and",
    "this",
    "that",
    "with",
    "you",
    "are",
    "for",
}
_TOKEN = re.compile(r"[\u3400-\u9fff]+|[a-z0-9_]+|[\U0001f300-\U0001faff\u2600-\u27bf]")


def tokenize(text: str) -> list[str]:
    terms = []
    for token in _TOKEN.findall(text[:MAX_INDEX_CHARS].casefold()):
        if "\u3400" <= token[0] <= "\u9fff":
            parts = (
                [token] if len(token) == 1 else [token[i : i + 2] for i in range(len(token) - 1)]
            )
        elif ord(token[0]) >= 0x2600:
            parts = ["emoji" + format(ord(token), "x")]
        else:
            parts = [token]
        terms.extend(part for part in parts if part not in _COMMON)
    return terms


def query_terms(text: str) -> list[str]:
    return list(dict.fromkeys(tokenize(text[-2000:])))[:MAX_QUERY_TERMS]


def alias_in_text(alias: str, text: str) -> bool:
    alias = alias.strip().casefold()
    text = text.casefold()
    if not alias:
        return False
    # A one-letter nickname such as K must not match every occurrence of "okay".
    if re.fullmatch(r"[a-z0-9_]+", alias):
        return re.search(r"(?<![a-z0-9_])" + re.escape(alias) + r"(?![a-z0-9_])", text) is not None
    return alias in text


def match_expression(terms: list[str]) -> str:
    # Terms are generated locally, and double quoting also neutralizes FTS syntax.
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def excerpt(text: str, terms: list[str], max_chars: int = MAX_EXCERPT_CHARS) -> str:
    """Keep the matching passage rather than blindly sending a long message head."""
    if len(text) <= max_chars:
        return text
    folded = text.casefold()
    positions = [position for term in terms if (position := folded.find(term)) >= 0]
    start = max(0, min(positions, default=0) - 80)
    start = min(start, max(0, len(text) - max_chars + 2))
    prefix = "…" if start else ""
    room = max_chars - len(prefix) - 1
    end = start + room
    return prefix + text[start:end] + ("…" if end < len(text) else "")


def bm25_fallback(documents: list[tuple[int, str]], terms: list[str]) -> list[tuple[int, float]]:
    """Score pre-tokenized documents without requiring an FTS extension."""
    if not documents or not terms:
        return []
    counters = [(sequence, Counter(tokens.split())) for sequence, tokens in documents]
    lengths = [sum(counts.values()) for _, counts in counters]
    average = sum(lengths) / max(len(lengths), 1) or 1
    frequencies = Counter(term for term in terms for _, counts in counters if term in counts)
    total = len(counters)
    weights = {
        term: math.log(1 + (total - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
        for term in terms
    }
    scores = []
    for (sequence, counts), length in zip(counters, lengths, strict=True):
        score = 0.0
        for term in terms:
            frequency = counts[term]
            if frequency:
                score += (
                    weights[term]
                    * frequency
                    * 2.2
                    / (frequency + 1.2 * (0.25 + 0.75 * length / average))
                )
        if score:
            scores.append((sequence, score))
    return sorted(scores, key=lambda item: item[1], reverse=True)
