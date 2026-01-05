import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
from model import GraphEncoder, GraphEncoderConfig
from utils import (
    load_id2emb,
    load_descriptions_from_graphs,
    PreprocessedGraphDataset,
    collate_fn,
)
import argparse


@torch.no_grad()
def retrieve_descriptions(
    model,
    train_graphs,
    test_graphs,
    train_emb_dict,
    device,
    output_csv,
    batch_size=64,
    top_k=5,
):

    # -------------------------
    # Load train descriptions
    # -------------------------
    train_id2desc = load_descriptions_from_graphs(train_graphs)

    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack([train_emb_dict[i] for i in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)

    print(f"Train set size: {len(train_ids)}")

    # -------------------------
    # Encode test graphs
    # -------------------------
    test_ds = PreprocessedGraphDataset(test_graphs)
    test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)



    test_embs = []
    test_ids_ordered = []
    ptr = 0

    for graphs in tqdm(test_dl, desc="Encoding test graphs"):
        graphs = graphs.to(device)
        z = model(graphs)
        z = F.normalize(z, dim=-1)
        test_embs.append(z)

        bs = graphs.num_graphs
        test_ids_ordered.extend(test_ds.ids[ptr:ptr + bs])
        ptr += bs

    test_embs = torch.cat(test_embs, dim=0)
    print(f"Encoded {test_embs.size(0)} test molecules")

    # -------------------------
    # Retrieval (k=1)
    # -------------------------
    K = top_k

    sims = test_embs @ train_embs.T
    topk_idx = sims.topk(K, dim=-1).indices.cpu()

    results = []
    for i, test_id in tqdm(
    enumerate(test_ids_ordered),
    total=len(test_ids_ordered),
    desc="Retrieving descriptions (top-k vote)",
):
        neighbor_ids = topk_idx[i].tolist()
        neighbor_train_ids = [train_ids[j] for j in neighbor_ids]
        neighbor_descs = [train_id2desc[j] for j in neighbor_train_ids]

        # Vote majoritaire
        desc_counter = Counter(neighbor_descs)
        voted_desc, _ = desc_counter.most_common(1)[0]

        results.append({
            "ID": test_id,
            "description": voted_desc,
        })

        if i < 5:
            print(f"\nTest {test_id}")
            for nid in neighbor_train_ids:
                print("  -", nid)
            print("→ Selected description:")
            print(voted_desc[:120], "...")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved submission to {output_csv}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Graph → text retrieval with top-k voting"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing train_graphs.pkl and test_graphs.pkl",
    )

    parser.add_argument(
        "--train_emb",
        type=str,
        required=True,
        help="CSV file with train text embeddings",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to trained GraphEncoder checkpoint",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV submission path",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for test graph encoding",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-k neighbors for retrieval voting",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Paths
    # -------------------------
    TRAIN_GRAPHS = os.path.join(args.data_dir, "train_graphs.pkl")
    TEST_GRAPHS = os.path.join(args.data_dir, "test_graphs.pkl")

    # -------------------------
    # Load text embeddings
    # -------------------------
    train_emb = load_id2emb(args.train_emb)
    emb_dim = len(next(iter(train_emb.values())))
    print(f"Loaded {len(train_emb)} train embeddings")

    # -------------------------
    # Load GraphEncoder
    # -------------------------
    cfg = GraphEncoderConfig(out_dim=emb_dim)
    model = GraphEncoder(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # -------------------------
    # Run retrieval
    # -------------------------
    retrieve_descriptions(
        model=model,
        train_graphs=TRAIN_GRAPHS,
        test_graphs=TEST_GRAPHS,
        train_emb_dict=train_emb,
        device=device,
        output_csv=args.output,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )



if __name__ == "__main__":
    main()
