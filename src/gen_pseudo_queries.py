import json

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.msmarco_utils import extract_docs_and_queries, load_jsonl


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    p, gpq = cfg.paths, cfg.gen_pseudo_queries

    print(f"Loading {gpq.model}...")
    tokenizer = AutoTokenizer.from_pretrained(gpq.model)
    model = T5ForConditionalGeneration.from_pretrained(
        gpq.model, torch_dtype=torch.float16
    ).to(gpq.device)
    model.eval()

    print("Loading MSMARCO docs...")
    raw = load_jsonl(p.msmarco_train)
    docs, _, data_format = extract_docs_and_queries(raw)
    print(f"Detected MSMARCO format: {data_format}")
    if not docs:
        raise ValueError(
            "No documents found in msmarco_train. Check dataset file path and format."
        )
    print(f"  {len(docs)} docs")

    with open(p.pseudo_queries, "w") as out_f:
        for i in tqdm(
            range(0, len(docs), gpq.batch_size), desc="Generating pseudo-queries"
        ):
            batch = docs[i : i + gpq.batch_size]
            texts = [d["text"] for d in batch]
            semids = [d["doc_id"] for d in batch]

            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=gpq.max_doc_len,
                return_tensors="pt",
            ).to(gpq.device)

            with torch.no_grad():
                outputs = model.generate(
                    **enc,
                    max_length=gpq.max_query_len,
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=gpq.n_queries,
                )

            # outputs: (batch * n_queries, seq_len)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for j, semid in enumerate(semids):
                pqs = decoded[j * gpq.n_queries : (j + 1) * gpq.n_queries]
                out_f.write(
                    json.dumps(
                        {"doc_idx": i + j, "doc_id": semid, "pseudo_queries": pqs}
                    )
                    + "\n"
                )

    print(f"Saved {len(docs)} entries → {p.pseudo_queries}")


if __name__ == "__main__":
    main()
