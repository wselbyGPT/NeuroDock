#!/usr/bin/env python3
"""
mono_sar_chembl.py

End-to-end:
- Pull ChEMBL bioactivity (human) for monoamine transporters + aminergic GPCRs
- Keep binding (B) + functional (F) assays with valid pChEMBL values
- Scaffold split by molecule (no leakage)
- Train a shared-trunk Keras regressor with TWO OUTPUT HEADS:
    (Morgan FP + target_id + standard_type) -> pChEMBL_B and pChEMBL_F
  where each head is trained with masked loss (missing labels don't contribute).
- Report RMSE for B and F heads overall + by target.
- Save test_predictions.csv with both-head predictions.

Notes:
- pChEMBL is -log10(molar); we filter standardized rows to nM and '='.
- assay_type "B"=binding, "F"=functional.
- standard_type includes Ki/Kd/IC50/EC50/etc; embedded as categorical input.
- Separate heads helps because B vs F are different modalities/noise regimes.

Requires:
  pip install pandas numpy tqdm scikit-learn tensorflow rdkit-pypi chembl_downloader
(or conda-forge RDKit preferred)

Example:
  python3 mono_sar_chembl.py --outdir mono_sar_out_twohead --epochs 40 --batch 512
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from sklearn.metrics import mean_squared_error

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator


# -----------------------------
# Defaults: human monoamine panel
# -----------------------------
DEFAULT_TARGETS = [
    # Transporters (human)
    "CHEMBL228",  # SERT / SLC6A4
    "CHEMBL238",  # DAT / SLC6A3
    "CHEMBL222",  # NET / SLC6A2
    # Aminergic GPCRs (human)
    "CHEMBL217",  # DRD2
    "CHEMBL214",  # HTR1A (5-HT1A)
    "CHEMBL224",  # HTR2A (5-HT2A)
]

# pChEMBL is defined for these standard types (per ChEMBL docs)
PCHEMBL_STANDARD_TYPES = ("IC50", "XC50", "EC50", "AC50", "Ki", "Kd", "Potency", "ED50")


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def get_table_columns(conn: sqlite3.Connection, table: str) -> set:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    return {row[1] for row in cur.fetchall()}


def connect_sqlite(path: str) -> sqlite3.Connection:
    if not os.path.exists(path):
        raise FileNotFoundError(f"ChEMBL SQLite not found: {path}")
    uri = f"file:{path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def fetch_chembl_df(
    conn: sqlite3.Connection,
    targets: List[str],
    min_confidence: int,
    assay_types: Tuple[str, ...] = ("B", "F"),
    organism: str = "Homo sapiens",
    single_protein_only: bool = True,
) -> pd.DataFrame:
    """
    Pull rows at the *measurement* level (one row per activity record)
    using parent molecule via molecule_hierarchy when available.

    Robust to older/newer ChEMBL SQLite schemas by guarding optional columns.
    """
    act_cols = get_table_columns(conn, "activities")
    assay_cols = get_table_columns(conn, "assays")

    where_parts = [
        "td.chembl_id IN ({})".format(",".join(["?"] * len(targets))),
        "td.organism = ?",
        "a.assay_type IN ({})".format(",".join(["?"] * len(assay_types))),
        "act.pchembl_value IS NOT NULL",
        "act.standard_relation = '='",
        "act.standard_units = 'nM'",
        "act.standard_type IN ({})".format(",".join(["?"] * len(PCHEMBL_STANDARD_TYPES))),
    ]

    params: List = []
    params.extend(targets)
    params.append(organism)
    params.extend(list(assay_types))
    params.extend(list(PCHEMBL_STANDARD_TYPES))

    has_conf = "confidence_score" in assay_cols
    if has_conf:
        where_parts.append("a.confidence_score >= ?")
        params.append(int(min_confidence))

    if single_protein_only:
        where_parts.append("td.target_type = 'SINGLE PROTEIN'")

    if "data_validity_comment" in act_cols:
        where_parts.append(
            "(act.data_validity_comment IS NULL OR act.data_validity_comment = 'Manually validated')"
        )
    if "potential_duplicate" in act_cols:
        where_parts.append("(act.potential_duplicate IS NULL OR act.potential_duplicate = 0)")

    where_sql = " AND ".join(where_parts)
    conf_select = "a.confidence_score AS confidence_score" if has_conf else "NULL AS confidence_score"

    sql = f"""
    SELECT
        mdp.chembl_id          AS molecule_chembl_id,
        cs.canonical_smiles    AS smiles,
        td.chembl_id           AS target_chembl_id,
        td.pref_name           AS target_name,
        a.assay_type           AS assay_type,
        {conf_select},
        act.standard_type      AS standard_type,
        act.pchembl_value      AS pchembl_value
    FROM activities act
    JOIN assays a ON act.assay_id = a.assay_id
    JOIN target_dictionary td ON a.tid = td.tid
    LEFT JOIN molecule_hierarchy mh ON act.molregno = mh.molregno
    JOIN molecule_dictionary mdp
        ON mdp.molregno = COALESCE(mh.parent_molregno, act.molregno)
    JOIN compound_structures cs
        ON cs.molregno = mdp.molregno
    WHERE {where_sql}
    ;
    """

    df = pd.read_sql(sql, conn, params=params)
    df = df.dropna(subset=["smiles", "pchembl_value", "target_chembl_id", "assay_type", "standard_type"])
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value"])
    return df


def canonicalize_smiles(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=True)


def murcko_scaffold_smiles(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(m)
    if scaf is None:
        return ""
    return Chem.MolToSmiles(scaf, isomericSmiles=False)


def greedy_scaffold_split(smiles: List[str], frac_val=0.15, frac_test=0.15, seed=0) -> Split:
    """
    Split by *molecule scaffold* (unique SMILES list).
    Greedy allocation by scaffold size (DeepChem-style-ish).
    """
    rng = np.random.default_rng(seed)

    scaff_to_indices: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles):
        sc = murcko_scaffold_smiles(smi)
        scaff_to_indices.setdefault(sc, []).append(i)

    scaff_items = list(scaff_to_indices.items())
    rng.shuffle(scaff_items)
    scaff_items.sort(key=lambda x: len(x[1]), reverse=True)

    n = len(smiles)
    n_test = int(round(frac_test * n))
    n_val = int(round(frac_val * n))

    test, val, train = [], [], []
    for _, idxs in scaff_items:
        if len(test) + len(idxs) <= n_test:
            test.extend(idxs)
        elif len(val) + len(idxs) <= n_val:
            val.extend(idxs)
        else:
            train.extend(idxs)

    return Split(
        train_idx=np.array(train, dtype=int),
        val_idx=np.array(val, dtype=int),
        test_idx=np.array(test, dtype=int),
    )


def morgan_fp_cache(unique_smiles: List[str], radius: int, n_bits: int) -> Dict[str, np.ndarray]:
    """
    Cache Morgan fingerprints per *unique* SMILES using modern RDKit generator API.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    cache: Dict[str, np.ndarray] = {}
    bad = 0

    for smi in tqdm(unique_smiles, desc="Fingerprints (unique SMILES)"):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            bad += 1
            continue
        bv = gen.GetFingerprint(m)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        cache[smi] = arr

    if bad:
        print(f"[warn] {bad} SMILES failed RDKit parse during fingerprinting.")
    return cache


def build_X_from_cache(rows_smiles: List[str], cache: Dict[str, np.ndarray], n_bits: int) -> np.ndarray:
    X = np.zeros((len(rows_smiles), n_bits), dtype=np.float32)
    misses = 0
    for i, smi in enumerate(rows_smiles):
        v = cache.get(smi)
        if v is None:
            misses += 1
            continue
        X[i, :] = v
    if misses:
        print(f"[warn] {misses} rows missing fingerprint in cache (will be zero vectors).")
    return X


def masked_mse(y_true, y_pred):
    """
    y_true is shape (batch, 2): [value, mask]
    y_pred is shape (batch, 1)
    """
    y = y_true[:, 0]
    m = y_true[:, 1]
    yhat = tf.squeeze(y_pred, axis=1)
    se = tf.square(y - yhat) * m
    denom = tf.reduce_sum(m) + 1e-8
    return tf.reduce_sum(se) / denom


def masked_rmse_metric(name: str):
    """
    Returns a Keras metric function that computes RMSE with the same [value, mask] encoding.
    """
    def _rmse(y_true, y_pred):
        y = y_true[:, 0]
        m = y_true[:, 1]
        yhat = tf.squeeze(y_pred, axis=1)
        se = tf.square(y - yhat) * m
        denom = tf.reduce_sum(m) + 1e-8
        mse = tf.reduce_sum(se) / denom
        return tf.sqrt(mse)
    _rmse.__name__ = name
    return _rmse


def build_twohead_model(
    fp_dim: int,
    n_targets: int,
    n_std_types: int,
    target_emb: int = 16,
    std_emb: int = 6,
) -> tf.keras.Model:
    """
    Shared trunk -> 2 heads:
      - out_B: predicted pChEMBL for binding
      - out_F: predicted pChEMBL for functional
    """
    fp_in = tf.keras.Input(shape=(fp_dim,), name="fp")
    t_in = tf.keras.Input(shape=(), dtype="int32", name="target_id")
    s_in = tf.keras.Input(shape=(), dtype="int32", name="standard_type")

    t_emb = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(n_targets, target_emb, name="target_emb")(t_in)
    )
    s_emb = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(n_std_types, std_emb, name="std_emb")(s_in)
    )

    x = tf.keras.layers.Concatenate()([fp_in, t_emb, s_emb])
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.10)(x)

    out_b = tf.keras.layers.Dense(1, name="out_B")(x)
    out_f = tf.keras.layers.Dense(1, name="out_F")(x)

    model = tf.keras.Model(inputs=[fp_in, t_in, s_in], outputs=[out_b, out_f])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"out_B": masked_mse, "out_F": masked_mse},
        metrics={"out_B": [masked_rmse_metric("rmse_B")], "out_F": [masked_rmse_metric("rmse_F")]},
    )
    return model


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def report_group_rmse_twohead(
    df_rows: pd.DataFrame,
    yb_true: np.ndarray,
    yb_pred: np.ndarray,
    mb: np.ndarray,
    yf_true: np.ndarray,
    yf_pred: np.ndarray,
    mf: np.ndarray,
    group_cols: List[str],
    title: str,
):
    tmp = df_rows.copy()

    tmp["yb_true"] = yb_true
    tmp["yb_pred"] = yb_pred
    tmp["mb"] = mb.astype(int)

    tmp["yf_true"] = yf_true
    tmp["yf_pred"] = yf_pred
    tmp["mf"] = mf.astype(int)

    def _rmse_masked(y, yhat, m):
        y = np.asarray(y, dtype=float)
        yhat = np.asarray(yhat, dtype=float)
        m = np.asarray(m, dtype=bool)
        if m.sum() == 0:
            return np.nan
        return float(np.sqrt(np.mean((y[m] - yhat[m]) ** 2)))

    rows = []
    for k, g in tmp.groupby(group_cols, dropna=False):
        r_b = _rmse_masked(g["yb_true"], g["yb_pred"], g["mb"])
        r_f = _rmse_masked(g["yf_true"], g["yf_pred"], g["mf"])
        n_b = int(g["mb"].sum())
        n_f = int(g["mf"].sum())
        rec = {}
        if len(group_cols) == 1:
            rec[group_cols[0]] = k
        else:
            rec.update(dict(zip(group_cols, k)))
        rec.update({"n_B": n_b, "rmse_B": r_b, "n_F": n_f, "rmse_F": r_f})
        rows.append(rec)

    out = pd.DataFrame(rows)
    print(f"\n== {title} ==")
    # sort by worst of the two (ignoring nan)
    out["worst"] = np.nanmax(out[["rmse_B", "rmse_F"]].to_numpy(dtype=float), axis=1)
    out = out.sort_values("worst", ascending=True).drop(columns=["worst"])
    print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chembl_sqlite", type=str, default="", help="Path to ChEMBL SQLite .db (optional).")
    ap.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS, help="Target ChEMBL IDs.")
    ap.add_argument("--min_confidence", type=int, default=7, help="Min assay confidence_score (if available).")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--n_bits", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_single_protein_only", action="store_true", help="Do not restrict to SINGLE PROTEIN targets.")
    ap.add_argument("--outdir", type=str, default="mono_sar_out_twohead")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Reproducibility (best-effort)
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # Optional: silence GPU warnings / force CPU (uncomment if desired)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if not args.chembl_sqlite:
        try:
            import chembl_downloader  # type: ignore
        except Exception as e:
            raise SystemExit("chembl_downloader not installed. Try: pip install chembl_downloader") from e
        print("[info] No --chembl_sqlite provided; downloading/extracting latest ChEMBL SQLite...")
        args.chembl_sqlite = chembl_downloader.download_extract_sqlite()

    print("[info] Using ChEMBL SQLite:", args.chembl_sqlite)

    with connect_sqlite(args.chembl_sqlite) as conn:
        df = fetch_chembl_df(
            conn,
            targets=list(args.targets),
            min_confidence=args.min_confidence,
            single_protein_only=(not args.no_single_protein_only),
        )

    # Canonicalize SMILES
    tqdm.pandas(desc="Canonical SMILES")
    df["smiles"] = df["smiles"].progress_apply(lambda s: canonicalize_smiles(s))
    df = df.dropna(subset=["smiles"])

    # Aggregate replicate measurements to median pChEMBL per (smiles, target, standard_type, assay_type)
    df = df.groupby(["smiles", "target_chembl_id", "standard_type", "assay_type"], as_index=False).agg(
        pchembl_value=("pchembl_value", "median"),
        target_name=("target_name", "first"),
    )

    print("\n[dataset long] rows:", len(df))
    print("[dataset long] unique molecules:", df["smiles"].nunique())
    print("[dataset long] assay_type counts:\n", df["assay_type"].value_counts())
    print("[dataset long] standard_type counts:\n", df["standard_type"].value_counts())

    # Pivot into "two-head" samples: one row per (smiles, target, standard_type)
    wide = df.pivot_table(
        index=["smiles", "target_chembl_id", "standard_type", "target_name"],
        columns="assay_type",
        values="pchembl_value",
        aggfunc="median",
    ).reset_index()

    # Build masks + fill missing labels with 0 (ignored by mask)
    wide["y_B"] = pd.to_numeric(wide.get("B"), errors="coerce")
    wide["y_F"] = pd.to_numeric(wide.get("F"), errors="coerce")
    wide["m_B"] = (~wide["y_B"].isna()).astype(np.float32)
    wide["m_F"] = (~wide["y_F"].isna()).astype(np.float32)
    wide["y_B"] = wide["y_B"].fillna(0.0).astype(np.float32)
    wide["y_F"] = wide["y_F"].fillna(0.0).astype(np.float32)

    # Some rows might have neither B nor F (shouldn't happen), but drop them
    wide = wide[(wide["m_B"] + wide["m_F"]) > 0].reset_index(drop=True)

    print("\n[dataset wide] rows:", len(wide))
    print("[dataset wide] B-labeled rows:", int(wide["m_B"].sum()))
    print("[dataset wide] F-labeled rows:", int(wide["m_F"].sum()))

    # Split on unique molecules (by scaffold)
    uniq_smiles = wide["smiles"].unique().tolist()
    split = greedy_scaffold_split(uniq_smiles, frac_val=0.15, frac_test=0.15, seed=args.seed)

    train_set = set(split.train_idx.tolist())
    val_set = set(split.val_idx.tolist())
    test_set = set(split.test_idx.tolist())

    smi_to_split: Dict[str, str] = {}
    for i, smi in enumerate(uniq_smiles):
        if i in train_set:
            smi_to_split[smi] = "train"
        elif i in val_set:
            smi_to_split[smi] = "val"
        else:
            smi_to_split[smi] = "test"

    wide["split"] = wide["smiles"].map(smi_to_split)
    train_df = wide[wide["split"] == "train"].reset_index(drop=True)
    val_df = wide[wide["split"] == "val"].reset_index(drop=True)
    test_df = wide[wide["split"] == "test"].reset_index(drop=True)

    print("\n[split sizes] train/val/test rows:", len(train_df), len(val_df), len(test_df))

    # Encode categoricals
    targets_sorted = sorted(wide["target_chembl_id"].unique().tolist())
    target_to_idx = {t: i for i, t in enumerate(targets_sorted)}
    std_sorted = sorted(wide["standard_type"].unique().tolist())
    std_to_idx = {s: i for i, s in enumerate(std_sorted)}

    # Fingerprint cache (unique SMILES in wide table)
    fp_cache = morgan_fp_cache(uniq_smiles, radius=args.radius, n_bits=args.n_bits)

    def featurize(rows: pd.DataFrame):
        smiles = rows["smiles"].tolist()
        X = build_X_from_cache(smiles, fp_cache, n_bits=args.n_bits)
        t_idx = rows["target_chembl_id"].map(target_to_idx).to_numpy(dtype=np.int32)
        s_idx = rows["standard_type"].map(std_to_idx).to_numpy(dtype=np.int32)

        # Pack y + mask for each head: shape (N, 2)
        yb = rows["y_B"].to_numpy(dtype=np.float32)
        mb = rows["m_B"].to_numpy(dtype=np.float32)
        yf = rows["y_F"].to_numpy(dtype=np.float32)
        mf = rows["m_F"].to_numpy(dtype=np.float32)

        yb_pack = np.stack([yb, mb], axis=1)
        yf_pack = np.stack([yf, mf], axis=1)
        return X, t_idx, s_idx, yb_pack, yf_pack, yb, mb, yf, mf

    Xtr, ttr, str_, yb_tr_pack, yf_tr_pack, yb_tr, mb_tr, yf_tr, mf_tr = featurize(train_df)
    Xva, tva, sva, yb_va_pack, yf_va_pack, yb_va, mb_va, yf_va, mf_va = featurize(val_df)
    Xte, tte, ste, yb_te_pack, yf_te_pack, yb_te, mb_te, yf_te, mf_te = featurize(test_df)

    model = build_twohead_model(
        fp_dim=args.n_bits,
        n_targets=len(targets_sorted),
        n_std_types=len(std_sorted),
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
    ]

    model.fit(
        {"fp": Xtr, "target_id": ttr, "standard_type": str_},
        {"out_B": yb_tr_pack, "out_F": yf_tr_pack},
        validation_data=(
            {"fp": Xva, "target_id": tva, "standard_type": sva},
            {"out_B": yb_va_pack, "out_F": yf_va_pack},
        ),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=2,
        callbacks=callbacks,
    )

    # Predict
    pred_b, pred_f = model.predict(
        {"fp": Xte, "target_id": tte, "standard_type": ste},
        batch_size=args.batch,
        verbose=0,
    )
    pred_b = pred_b.reshape(-1).astype(np.float32)
    pred_f = pred_f.reshape(-1).astype(np.float32)

    # Compute masked RMSE per head
    def _rmse_masked(y, yhat, m):
        m = (m.astype(np.float32) > 0.5)
        if m.sum() == 0:
            return float("nan")
        return float(np.sqrt(np.mean((y[m] - yhat[m]) ** 2)))

    rmse_b = _rmse_masked(yb_te, pred_b, mb_te)
    rmse_f = _rmse_masked(yf_te, pred_f, mf_te)
    print("\n[Test] RMSE_B:", rmse_b)
    print("[Test] RMSE_F:", rmse_f)

    report_group_rmse_twohead(
        test_df,
        yb_true=yb_te, yb_pred=pred_b, mb=mb_te,
        yf_true=yf_te, yf_pred=pred_f, mf=mf_te,
        group_cols=["target_chembl_id"],
        title="Test RMSE by target (B and F heads)",
    )

    # Save predictions
    out = test_df.copy()
    out["yB_true"] = yb_te
    out["mB"] = mb_te
    out["yF_true"] = yf_te
    out["mF"] = mf_te
    out["yB_pred"] = pred_b
    out["yF_pred"] = pred_f

    out_csv = os.path.join(args.outdir, "test_predictions.csv")
    out.to_csv(out_csv, index=False)
    print("\n[Saved]", out_csv)


if __name__ == "__main__":
    main()
