#!/usr/bin/env python3
"""
viz_mono_twohead.py

Visual diagnostics for the TWO-HEAD mono SAR model output.

Input: test_predictions.csv produced by mono_sar_chembl.py (two-head version),
with columns:
  smiles,target_chembl_id,standard_type,target_name,split,
  yB_true,mB,yB_pred,
  yF_true,mF,yF_pred

Outputs (in --save_dir):
  - parity_B.png / parity_F.png
  - residual_hist_B.png / residual_hist_F.png
  - residual_vs_true_B.png / residual_vs_true_F.png
  - rmse_by_target_B.png / rmse_by_target_F.png
  - metrics_summary.csv
  - top_outliers_B.csv / top_outliers_F.csv
  - optional UMAP: umap_abs_error_B.png / umap_abs_error_F.png

No seaborn. Matplotlib only.

Example:
  python3 viz_mono_twohead.py --pred_csv mono_sar_out_twohead/test_predictions.csv --save_dir mono_sar_out_twohead/figs --hexbin
  python3 viz_mono_twohead.py --pred_csv mono_sar_out_twohead/test_predictions.csv --save_dir mono_sar_out_twohead/figs --umap
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rmse(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(y - yhat)))


def bias(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(yhat - y))  # + means overpredict


def r2(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def masked_arrays(df: pd.DataFrame, head: str):
    """
    head: "B" or "F"
    Returns y, yhat, resid, abs_err on masked rows only.
    """
    y_col = f"y{head}_true"
    p_col = f"y{head}_pred"
    m_col = f"m{head}"

    d = df.copy()
    for c in [y_col, p_col, m_col]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=[p_col, m_col])
    d = d[d[m_col] > 0.5].reset_index(drop=True)
    d = d.dropna(subset=[y_col]).reset_index(drop=True)

    d["resid"] = d[y_col] - d[p_col]
    d["abs_err"] = np.abs(d["resid"])
    return d


def metrics_table(d: pd.DataFrame, head: str) -> pd.DataFrame:
    y_col = f"y{head}_true"
    p_col = f"y{head}_pred"
    y = d[y_col].to_numpy()
    yhat = d[p_col].to_numpy()
    return pd.DataFrame([{
        "head": head,
        "n": int(len(d)),
        "rmse": rmse(y, yhat),
        "mae": mae(y, yhat),
        "bias": bias(y, yhat),
        "r2": r2(y, yhat),
    }])


def group_rmse(d: pd.DataFrame, head: str, group_cols):
    y_col = f"y{head}_true"
    p_col = f"y{head}_pred"

    rows = []
    for k, g in d.groupby(group_cols, dropna=False):
        y = g[y_col].to_numpy()
        yhat = g[p_col].to_numpy()
        rec = {"head": head, "n": int(len(g)), "rmse": rmse(y, yhat)}
        if len(group_cols) == 1:
            rec[group_cols[0]] = k
        else:
            rec.update(dict(zip(group_cols, k)))
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("rmse", ascending=True)
    return out


def parity_plot(dplot: pd.DataFrame, head: str, save_path: str, hexbin: bool):
    y_col = f"y{head}_true"
    p_col = f"y{head}_pred"

    y = dplot[y_col].to_numpy()
    yhat = dplot[p_col].to_numpy()
    r = rmse(y, yhat)

    plt.figure()
    if hexbin:
        plt.hexbin(dplot[y_col], dplot[p_col], gridsize=60)
        plt.colorbar(label="count")
    else:
        plt.scatter(dplot[y_col], dplot[p_col], s=8, alpha=0.35)

    mn = float(min(y.min(), yhat.min()))
    mx = float(max(y.max(), yhat.max()))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel(f"True pChEMBL ({head})")
    plt.ylabel(f"Predicted pChEMBL ({head})")
    plt.title(f"Parity {head} (RMSE={r:.3f})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)


def residual_hist(d: pd.DataFrame, head: str, save_path: str):
    plt.figure()
    plt.hist(d["resid"], bins=60)
    plt.xlabel(f"Residual (true - pred) [{head}]")
    plt.ylabel("Count")
    plt.title(f"Residual distribution {head}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)


def residual_vs_true(dplot: pd.DataFrame, head: str, save_path: str):
    y_col = f"y{head}_true"
    plt.figure()
    plt.scatter(dplot[y_col], dplot["resid"], s=8, alpha=0.35)
    plt.axhline(0.0)
    plt.xlabel(f"True pChEMBL ({head})")
    plt.ylabel("Residual (true - pred)")
    plt.title(f"Residual vs True ({head})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)


def rmse_bar_by_target(grp: pd.DataFrame, head: str, save_path: str):
    # expects columns: target_chembl_id, rmse
    labels = grp["target_chembl_id"].astype(str).tolist()
    x = np.arange(len(labels))

    plt.figure(figsize=(max(8, len(labels) * 0.7), 4))
    plt.bar(x, grp["rmse"].to_numpy())
    plt.xticks(x, labels, rotation=0)
    plt.ylabel(f"RMSE pChEMBL ({head})")
    plt.title(f"RMSE by target ({head})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)


def do_umap(d: pd.DataFrame, head: str, save_path: str, max_points: int, seed: int):
    try:
        import umap
        from rdkit import Chem
        from rdkit import DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except Exception as e:
        raise SystemExit("For --umap you need: pip install umap-learn and RDKit installed.") from e

    dd = d.copy()
    if len(dd) > max_points:
        dd = dd.sample(max_points, random_state=seed).reset_index(drop=True)

    # FP once per unique SMILES
    uniq = dd["smiles"].astype(str).unique().tolist()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    fp_map = {}
    bad = 0
    for smi in uniq:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            bad += 1
            continue
        bv = gen.GetFingerprint(m)
        arr = np.zeros((2048,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        fp_map[smi] = arr

    if bad:
        print(f"[warn] {bad} SMILES failed RDKit parse for UMAP.")

    keep = dd["smiles"].map(lambda s: s in fp_map).to_numpy()
    dd = dd.iloc[keep].reset_index(drop=True)
    X = np.stack([fp_map[s] for s in dd["smiles"].tolist()], axis=0).astype(bool)

    reducer = umap.UMAP(n_neighbors=25, min_dist=0.15, metric="jaccard", random_state=seed)
    emb = reducer.fit_transform(X)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=dd["abs_err"].to_numpy(), s=8, alpha=0.7)
    plt.colorbar(sc, label=f"|error| pChEMBL ({head})")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(f"Chemical space UMAP colored by |error| ({head})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", default="mono_sar_out_twohead/test_predictions.csv")
    ap.add_argument("--save_dir", default="mono_sar_out_twohead/figs")
    ap.add_argument("--max_points", type=int, default=200000, help="Cap points for scatter-style plots.")
    ap.add_argument("--hexbin", action="store_true", help="Use hexbin for parity plot (recommended if many points).")
    ap.add_argument("--outliers", type=int, default=200, help="Save top-N absolute-error rows to CSV per head.")
    ap.add_argument("--umap", action="store_true", help="Also make chemical-space UMAP colored by |error|.")
    ap.add_argument("--umap_max", type=int, default=15000, help="Max points for UMAP.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    df = pd.read_csv(args.pred_csv)

    # Validate expected columns
    need = [
        "smiles", "target_chembl_id", "standard_type", "split",
        "yB_true", "mB", "yB_pred",
        "yF_true", "mF", "yF_pred",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {args.pred_csv}: {missing}\nFound: {list(df.columns)}")

    # Build masked frames
    dB = masked_arrays(df, "B")
    dF = masked_arrays(df, "F")

    print(f"[info] B rows (masked): {len(dB):,}")
    print(f"[info] F rows (masked): {len(dF):,}")

    # Metrics
    overall = pd.concat([metrics_table(dB, "B"), metrics_table(dF, "F")], ignore_index=True)
    by_t_B = group_rmse(dB, "B", ["target_chembl_id"])
    by_t_F = group_rmse(dF, "F", ["target_chembl_id"])

    metrics_path = os.path.join(args.save_dir, "metrics_summary.csv")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("# overall\n")
        overall.to_csv(f, index=False)
        f.write("\n# by_target_B\n")
        by_t_B.to_csv(f, index=False)
        f.write("\n# by_target_F\n")
        by_t_F.to_csv(f, index=False)
    print("Saved", metrics_path)

    # Outliers per head
    outB = dB.sort_values("abs_err", ascending=False).head(args.outliers)
    outF = dF.sort_values("abs_err", ascending=False).head(args.outliers)
    outB_path = os.path.join(args.save_dir, f"top_{args.outliers}_outliers_B.csv")
    outF_path = os.path.join(args.save_dir, f"top_{args.outliers}_outliers_F.csv")
    outB.to_csv(outB_path, index=False)
    outF.to_csv(outF_path, index=False)
    print("Saved", outB_path)
    print("Saved", outF_path)

    # Downsample for plotting speed
    dB_plot = dB
    dF_plot = dF
    if len(dB_plot) > args.max_points:
        dB_plot = dB_plot.sample(args.max_points, random_state=args.seed).reset_index(drop=True)
    if len(dF_plot) > args.max_points:
        dF_plot = dF_plot.sample(args.max_points, random_state=args.seed).reset_index(drop=True)

    # Parity
    parity_plot(dB_plot, "B", os.path.join(args.save_dir, "parity_B.png"), args.hexbin)
    print("Saved", os.path.join(args.save_dir, "parity_B.png"))
    parity_plot(dF_plot, "F", os.path.join(args.save_dir, "parity_F.png"), args.hexbin)
    print("Saved", os.path.join(args.save_dir, "parity_F.png"))

    # Residual hist
    residual_hist(dB, "B", os.path.join(args.save_dir, "residual_hist_B.png"))
    print("Saved", os.path.join(args.save_dir, "residual_hist_B.png"))
    residual_hist(dF, "F", os.path.join(args.save_dir, "residual_hist_F.png"))
    print("Saved", os.path.join(args.save_dir, "residual_hist_F.png"))

    # Residual vs true
    residual_vs_true(dB_plot, "B", os.path.join(args.save_dir, "residual_vs_true_B.png"))
    print("Saved", os.path.join(args.save_dir, "residual_vs_true_B.png"))
    residual_vs_true(dF_plot, "F", os.path.join(args.save_dir, "residual_vs_true_F.png"))
    print("Saved", os.path.join(args.save_dir, "residual_vs_true_F.png"))

    # RMSE by target bars
    rmse_bar_by_target(by_t_B, "B", os.path.join(args.save_dir, "rmse_by_target_B.png"))
    print("Saved", os.path.join(args.save_dir, "rmse_by_target_B.png"))
    rmse_bar_by_target(by_t_F, "F", os.path.join(args.save_dir, "rmse_by_target_F.png"))
    print("Saved", os.path.join(args.save_dir, "rmse_by_target_F.png"))

    # Optional UMAP
    if args.umap:
        do_umap(dB, "B", os.path.join(args.save_dir, "umap_abs_error_B.png"), args.umap_max, args.seed)
        print("Saved", os.path.join(args.save_dir, "umap_abs_error_B.png"))
        do_umap(dF, "F", os.path.join(args.save_dir, "umap_abs_error_F.png"), args.umap_max, args.seed)
        print("Saved", os.path.join(args.save_dir, "umap_abs_error_F.png"))

    print("\nDone. Outputs are in:", args.save_dir)


if __name__ == "__main__":
    main()
