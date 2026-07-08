import json
from pathlib import Path


SRC_DIR = Path("/home/guest/r14944001/projects/GenRet/dataset/nq320k")
OUT_DIR = Path("data/nq320k/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def write_split(out_path: Path, corpus: list[str], queries: list[list]):
    with out_path.open("w") as out:
        # 1. write corpus documents
        for doc_id, text in enumerate(corpus):
            out.write(json.dumps({
                "operation": "indexing",
                "doc_id": str(doc_id),
                "text": text,
            }, ensure_ascii=False) + "\n")

        # 2. write query -> positive doc mappings
        for item in queries:
            query_text, doc_id = item
            out.write(json.dumps({
                "operation": "query",
                "doc_id": str(doc_id),
                "text": query_text,
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path}")
    print(f"  docs: {len(corpus)}")
    print(f"  queries: {len(queries)}")
    print(f"  total lines: {len(corpus) + len(queries)}")


def main():
    corpus = load_json(SRC_DIR / "corpus_lite.json")
    train = load_json(SRC_DIR / "train.json")
    dev = load_json(SRC_DIR / "dev.json")

    write_split(OUT_DIR / "train.jsonl", corpus, train)
    write_split(OUT_DIR / "valid.jsonl", corpus, [])
    write_split(OUT_DIR / "test.jsonl", corpus, dev)



if __name__ == "__main__":
    main()