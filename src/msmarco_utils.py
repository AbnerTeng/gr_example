import hashlib
import json
from typing import Any, Dict, List, Tuple


JsonDict = Dict[str, Any]


def load_jsonl(path: str) -> List[JsonDict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def detect_msmarco_format(raw: List[JsonDict]) -> str:
    for row in raw:
        if not isinstance(row, dict):
            continue
        if "operation" in row:
            return "operation"
        if "query" in row and "passages" in row:
            return "qa"
    return "unknown"


def _is_positive(flag: Any) -> bool:
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, (int, float)):
        return int(flag) == 1
    if isinstance(flag, str):
        return flag.strip() in {"1", "true", "True"}
    return False


def _build_doc_id(text: str, url: str | None = None) -> str:
    key = f"{url or ''}\n{text.strip()}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"doc_{digest}"


def extract_docs_and_queries(
    raw: List[JsonDict],
) -> Tuple[List[JsonDict], List[JsonDict], str]:
    fmt = detect_msmarco_format(raw)
    docs: Dict[str, JsonDict] = {}
    queries: List[JsonDict] = []

    if fmt == "operation":
        for row in raw:
            op = row.get("operation")
            doc_id = row.get("doc_id")
            text = row.get("text")
            if not isinstance(doc_id, str) or not isinstance(text, str):
                continue
            if op == "indexing":
                if doc_id not in docs:
                    docs[doc_id] = {"doc_id": doc_id, "text": text}
            elif op == "query":
                queries.append({"doc_id": doc_id, "text": text})
        return list(docs.values()), queries, fmt

    if fmt == "qa":
        for row in raw:
            query_text = row.get("query")
            if not isinstance(query_text, str) or not query_text.strip():
                continue

            passages = row.get("passages") or {}
            passage_texts = passages.get("passage_text") or []
            is_selected = passages.get("is_selected") or []
            urls = passages.get("url") or []

            for i, passage_text in enumerate(passage_texts):
                if i >= len(is_selected) or not _is_positive(is_selected[i]):
                    continue
                if not isinstance(passage_text, str) or not passage_text.strip():
                    continue
                url = urls[i] if i < len(urls) and isinstance(urls[i], str) else None

                doc_id = _build_doc_id(passage_text, url)
                if doc_id not in docs:
                    docs[doc_id] = {"doc_id": doc_id, "text": passage_text}
                queries.append({"doc_id": doc_id, "text": query_text.strip()})

        return list(docs.values()), queries, fmt

    return [], [], fmt
