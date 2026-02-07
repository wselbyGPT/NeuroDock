#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", default="mono_sar_out/test_predictions.csv")
    ap.add_argument("--save_dir", default="mono_sar_out/figs")
    ap.add_argument("--umap", action="store_true", help="Also make chemical-space UMAP (needs umap-learn + rdkit).")
    args = ap.parse_args()

    import os
    os.makedirs(args.save_dir, exist_ok=True)

    df = pd.read_csv(args.pred_csv)
    # expected columns from the training script:
    # smiles,target_chembl_id,assay_type,pchembl_value,target_name,split,y_true,y_pred
    for col in ["y_true", "y_pred", "target_chembl_id", "assay_type"]:
        if col not in df.columns:
            raise SystemExit(f"Missing column {col} in {args.pred_csv}. Found: {list(df.columns)}")

    df["resid"] = df["y_true"] - df["y_pred"]
    df["abs_err"] = np.abs(df["resid"])

    # -----------------------
    # Plot 1: True vs Predicted (overall)
    # -----------------------
    y = df["y_true"].to_numpy()
    yhat = df["y_pred"].to_numpy()
    r = rmse(y, yhat)

    plt.figure()
    plt.scatter(y, yhat, s=8, alpha=0.35)
    mn = min(y.min(), yhat.min())
    mx = max(y.max(), yhat.max())
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True pChEMBL")
    plt.ylabel("Predicted pChEMBL")
    plt.title(f"Test: True vs Predicted (RMSE={r:.3f})")
    out1 = os.path.join(args.save_dir, "true_vs_pred_overall.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    print("Saved", out1)

    # -----------------------
    # Plot 2: Residual histogram
    # -----------------------
    plt.figure()
    plt.hist(df["resid"], bins=60)
    plt.xlabel("Residual (true - pred)")
    plt.ylabel("Count")
    plt.title("Test residual distribution")
    out2 = os.path.join(args.save_dir, "residual_hist.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    print("Saved", out2)

    # -----------------------
    # Plot 3: RMSE by target and assay type
    # -----------------------
    # compute group RMSE
    grp = (
        df.groupby(["target_chembl_id", "assay_type"])
          .apply(lambda g: rmse(g["y_true"].to_numpy(), g["y_pred"].to_numpy()))
          .reset_index(name="rmse")
    )

    # simple bar-like plot without seaborn
    # build x labels and positions
    labels = [f"{t}\n{a}" for t, a in zip(grp["target_chembl_id"], grp["assay_type"])]
    x = np.arange(len(labels))

    plt.figure(figsize=(max(8, len(labels) * 0.7), 4))
    plt.bar(x, grp["rmse"].to_numpy())
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("RMSE (pChEMBL)")
    plt.title("Test RMSE by Target + Assay Type")
    out3 = os.path.join(args.save_dir, "rmse_by_target_assay.png")
    plt.tight_layout()
    plt.savefig(out3, dpi=200)
    print("Saved", out3)

    # -----------------------
    # Bonus: Chemical space UMAP (colored by absolute error)
    # -----------------------
    if args.umap:
        try:
            import umap
            from rdkit import Chem
            from rdkit import DataStructs
            from rdkit.Chem import AllChem
        except Exception as e:
            raise SystemExit("For --umap you need: pip install umap-learn and RDKit installed.") from e

        def fp(smi, n_bits=2048, radius=2):
            m = Chem.MolFromSmiles(smi)
            if m is None:
                return None
            bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv, arr)
            return arr

        # sample if huge
        d = df.copy()
        if len(d) > 15000:
            d = d.sample(15000, random_state=0).reset_index(drop=True)

        X = []
        keep = []
        for i, smi in enumerate(d["smiles"].astype(str).tolist()):
            a = fp(smi)
            if a is not None:
                X.append(a)
                keep.append(i)

        d = d.iloc[keep].reset_index(drop=True)
        X = np.asarray(X, dtype=np.float32)

        reducer = umap.UMAP(n_neighbors=25, min_dist=0.15, metric="euclidean", random_state=0)
        emb = reducer.fit_transform(X)

        plt.figure(figsize=(6, 5))
        plt.scatter(emb[:, 0], emb[:, 1], c=d["abs_err"].to_numpy(), s=8, alpha=0.7)
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.title("Chemical space (UMAP), colored by |error|")
        out4 = os.path.join(args.save_dir, "umap_abs_error.png")
        plt.tight_layout()
        plt.savefig(out4, dpi=200)
        print("Saved", out4)

    print("\nDone. Images are in:", args.save_dir)

if __name__ == "__main__":
    main()
