#!/usr/bin/env python3
# eval_bleu.py
# --------------------------------------------------
# Simple BLEU evaluation script (sacreBLEU)
# --------------------------------------------------

import argparse
from pathlib import Path

import pandas as pd
import sacrebleu

from utils import load_descriptions_from_graphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True,
                        help="CSV with columns: ID, description")
    parser.add_argument("--ref_graphs", type=str, required=True,
                        help="Path to *_graphs.pkl containing reference descriptions")
    args = parser.parse_args()

    # -------------------------------
    # Load predictions
    # -------------------------------
    preds_df = pd.read_csv(args.pred_csv)
    assert {"ID", "description"} <= set(preds_df.columns)

    id2pred = dict(zip(preds_df["ID"], preds_df["description"]))

    # -------------------------------
    # Load references
    # -------------------------------
    id2ref = load_descriptions_from_graphs(args.ref_graphs)

    # Align predictions & references
    preds = []
    refs = []

    missing = 0
    for id_, pred in id2pred.items():
        if id_ not in id2ref:
            missing += 1
            continue
        preds.append(str(pred))
        refs.append(str(id2ref[id_]))

    print(f"Aligned pairs: {len(preds)}")
    if missing > 0:
        print(f"Warning: {missing} IDs not found in references")

    # -------------------------------
    # BLEU computation
    # -------------------------------
    bleu = sacrebleu.corpus_bleu(
        preds,
        [refs],            # list of reference lists
        tokenize="13a"
    )

    print("\n================ BLEU =================")
    print(f"BLEU score: {bleu.score:.2f}")
    print("======================================\n")


if __name__ == "__main__":
    main()
