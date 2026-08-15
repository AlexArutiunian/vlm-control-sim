# csv_similarity.py
import argparse
import sys
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_FILENAME = "command.csv"

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # basic validation
    need_cols = {"id", "command"}
    missing = need_cols - set(df.columns.str.lower())
    # if headers differ only in case — normalize them
    cols_map = {c: c.lower() for c in df.columns}
    df.rename(columns=cols_map, inplace=True)
    if not need_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {need_cols}. Found: {list(df.columns)}")
    # ensure command column is string
    df["command"] = df["command"].astype(str)
    return df[["id", "command"]].copy()

def build_model(name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(name)

def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    emb = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return emb.astype(np.float32)

def compute_similarities(user_cmd: str, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    model = build_model(model_name)
    corpus_emb = embed_texts(model, df["command"].tolist())
    query_emb = embed_texts(model, [user_cmd])[0]
    sims = corpus_emb @ query_emb  # normalized => cosine similarity
    out = df.copy()
    out["sim"] = sims
    out.sort_values("sim", ascending=False, inplace=True, kind="mergesort")
    # optionally round for display, but keep full precision
    return out

def main():
    ap = argparse.ArgumentParser(
        description="Compute similarity between a command and a library (CSV -> command.csv)."
    )
    ap.add_argument("--input", required=True, help="Path to input CSV with columns id,command")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformers model name")
    ap.add_argument("--query", help="User command. If omitted — interactive input will be used")
    args = ap.parse_args()

    df = load_csv(args.input)

    user_cmd = args.query
    if not user_cmd:
        try:
            user_cmd = input("Enter command: ").strip()
        except KeyboardInterrupt:
            print("\nCancelled.", file=sys.stderr)
            sys.exit(1)

    if not user_cmd:
        print("Empty command — nothing to compare.", file=sys.stderr)
        sys.exit(2)

    result = compute_similarities(user_cmd, df, args.model)

    # Save to 'command.csv' in UTF-8-SIG (Excel-friendly), comma separator, with headers
    result.to_csv(OUTPUT_FILENAME, index=False, encoding="utf-8-sig")

    # Short console report
    print(f"Done. Saved: {OUTPUT_FILENAME}")
    # Show top-5 for quick inspection
    preview = result.head(5).copy()
    preview["sim"] = preview["sim"].round(3)
    print(preview.to_string(index=False))

if __name__ == "__main__":
    main()
