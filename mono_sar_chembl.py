#!/usr/bin/env python3
"""
mono_sar_chembl.py

End-to-end:
- Pull ChEMBL bioactivity (human) for monoamine transporters + aminergic GPCRs
- Keep binding (B) + functional (F) assays with valid pChEMBL values
- Scaffold split by molecule (no leakage)
- Train conditional Keras regressor: (Morgan FP + target_id + assay_type) -> pChEMBL
- Report RMSE overall + by target + by assay_type

Notes:
- pChEMBL is -log10(molar) for certain standard types and only when standardized to nM and "=".
- assay_type "B"=binding, "F"=functional.
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
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


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


def get_activity_columns(conn: sqlite3.Connection) -> set:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(activities);")
    cols = {row[1] for row in cur.fetchall()}  # name in second field
    return cols


def get_assay_columns(conn: sqlite3.Connection) -> set:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(assays);")
    cols = {row[1] for row in cur.fetchall()}
    return cols


def connect_sqlite(path: str) -> sqlite3.Connection:
    if not os.path.exists(path):
        raise FileNotFoundError(f"ChEMBL SQLite not found: {path}")
    # read-only is safer for huge files
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
    Pulls rows at the *measurement* level (one row per activity record)
    using parent molecule via molecule_hierarchy when available.
    """
    act_cols = get_activity_columns(conn)
    assay_cols = get_assay_columns(conn)

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

    if "confidence_score" in assay_cols:
        where_parts.append("a.confidence_score >= ?")
        params.append(int(min_confidence))

    if single_protein_only:
        where_parts.append("td.target_type = 'SINGLE PROTEIN'")

    # Optional quality columns (present in modern ChEMBL)
    if "data_validity_comment" in act_cols:
        where_parts.append("(act.data_validity_comment IS NULL OR act.data_validity_comment = 'Manually validated')")

    if "potential_duplicate" in act_cols:
        where_parts.append("(act.potential_duplicate IS NULL OR act.potential_duplicate = 0)")

    where_sql = " AND ".join(where_parts)

    # Use parent molecules when possible to avoid salt duplicates
    sql = f"""
    SELECT
        mdp.chembl_id          AS molecule_chembl_id,
        cs.canonical_smiles    AS smiles,
        td.chembl_id           AS target_chembl_id,
        td.pref_name           AS target_name,
        a.assay_type           AS assay_type,
        a.confidence_score     AS confidence_score,
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
    df = df.dropna(subset=["smiles", "pchembl_value", "target_chembl_id", "assay_type"])
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

    # sort scaffolds by descending size, but shuffle ties for a bit of randomness
    scaff_items = list(scaff_to_indices.items())
    rng.shuffle(scaff_items)
    scaff_items.sort(key=lambda x: len(x[1]), reverse=True)

    n = len(smiles)
    n_test = int(round(frac_test * n))
    n_val = int(round(frac_val * n))

    test, val, train = [], [], []
    for sc, idxs in scaff_items:
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


def morgan_fp_bits(smiles: List[str], radius: int, n_bits: int) -> np.ndarray:
    X = np.zeros((len(smiles), n_bits), dtype=np.float32)
    bad = 0
    for i, smi in enumerate(tqdm(smiles, desc="Fingerprints")):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            bad += 1
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        X[i, :] = arr
    if bad:
        print(f"[warn] {bad} SMILES failed RDKit parse during fingerprinting.")
    return X


def build_model(fp_dim: int, n_targets: int, n_assay_types: int, target_emb=16, assay_emb=4) -> tf.keras.Model:
    fp_in = tf.keras.Input(shape=(fp_dim,), name="fp")
    t_in = tf.keras.Input(shape=(), dtype="int32", name="target_id")
    a_in = tf.keras.Input(shape=(), dtype="int32", name="assay_type")

    t_emb = tf.keras.layers.Embedding(n_targets, target_emb, name="target_emb")(t_in)
    a_emb = tf.keras.layers.Embedding(n_assay_types, assay_emb, name="assay_emb")(a_in)

    t_emb = tf.keras.layers.Flatten()(t_emb)
    a_emb = tf.keras.layers.Flatten()(a_emb)

    x = tf.keras.layers.Concatenate()([fp_in, t_emb, a_emb])
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.10)(x)
    out = tf.keras.layers.Dense(1, name="pchembl")(x)

    model = tf.keras.Model([fp_in, t_in, a_in], out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def report_group_rmse(df_rows: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, group_cols: List[str], title: str):
    tmp = df_rows.copy()
    tmp["y_true"] = y_true
    tmp["y_pred"] = y_pred
    tmp["se"] = (tmp["y_true"] - tmp["y_pred"]) ** 2

    grp = tmp.groupby(group_cols)["se"].mean().reset_index()
    grp["rmse"] = np.sqrt(grp["se"])
    grp = grp.sort_values("rmse", ascending=True)

    print(f"\n== {title} ==")
    print(grp[group_cols + ["rmse"]].to_string(index=False))


def top_bits_via_gradients(
    model: tf.keras.Model,
    X: np.ndarray,
    t_idx: np.ndarray,
    a_idx: np.ndarray,
    n_top: int = 30,
    max_samples: int = 4096,
    seed: int = 0,
) -> List[Tuple[int, float]]:
    """
    Very rough global importance: mean(|grad * input|) across samples.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    take = min(n, max_samples)
    sel = rng.choice(n, size=take, replace=False)

    Xs = tf.convert_to_tensor(X[sel], dtype=tf.float32)
    ts = tf.convert_to_tensor(t_idx[sel], dtype=tf.int32)
    a_s = tf.convert_to_tensor(a_idx[sel], dtype=tf.int32)

    with tf.GradientTape() as tape:
        tape.watch(Xs)
        yhat = model([Xs, ts, a_s], training=False)
        yhat = tf.squeeze(yhat, axis=1)

    grads = tape.gradient(yhat, Xs).numpy()
    scores = np.mean(np.abs(grads * X[sel]), axis=0)
    top = np.argsort(-scores)[:n_top]
    return [(int(i), float(scores[i])) for i in top]


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
    ap.add_argument("--dump_bits", action="store_true", help="Compute and save rough top fingerprint-bit importances.")
    ap.add_argument("--outdir", type=str, default="mono_sar_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Download ChEMBL SQLite if not provided
    if not args.chembl_sqlite:
        try:
            import chembl_downloader  # type: ignore
        except Exception as e:
            raise SystemExit("chembl_downloader not installed. Try: pip install chembl_downloader") from e
        print("[info] No --chembl_sqlite provided; downloading/extracting latest ChEMBL SQLite...")
        args.chembl_sqlite = chembl_downloader.download_extract_sqlite()  # returns path to .db

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

    # Aggregate replicate measurements: median pChEMBL per (molecule, target, assay_type)
    df = df.groupby(["smiles", "target_chembl_id", "assay_type"], as_index=False).agg(
        pchembl_value=("pchembl_value", "median"),
        target_name=("target_name", "first"),
    )

    print("\n[dataset] rows:", len(df))
    print("[dataset] unique molecules:", df["smiles"].nunique())
    print("[dataset] targets:", sorted(df["target_chembl_id"].unique().tolist()))
    print("[dataset] assay_type counts:\n", df["assay_type"].value_counts())

    # Build molecule list and scaffold split on molecules (prevents leakage across targets/assay_types)
    uniq_smiles = df["smiles"].unique().tolist()
    split = greedy_scaffold_split(uniq_smiles, frac_val=0.15, frac_test=0.15, seed=args.seed)

    smi_to_split = {}
    for i, smi in enumerate(uniq_smiles):
        if i in set(split.train_idx):
            smi_to_split[smi] = "train"
        elif i in set(split.val_idx):
            smi_to_split[smi] = "val"
        else:
            smi_to_split[smi] = "test"

    df["split"] = df["smiles"].map(smi_to_split)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Encode categorical inputs
    targets_sorted = sorted(df["target_chembl_id"].unique().tolist())
    target_to_idx = {t: i for i, t in enumerate(targets_sorted)}
    assay_to_idx = {"B": 0, "F": 1}

    def featurize_rows(rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        smiles = rows["smiles"].tolist()
        X = morgan_fp_bits(smiles, radius=args.radius, n_bits=args.n_bits)
        t_idx = rows["target_chembl_id"].map(target_to_idx).to_numpy(dtype=np.int32)
        a_idx = rows["assay_type"].map(assay_to_idx).to_numpy(dtype=np.int32)
        y = rows["pchembl_value"].to_numpy(dtype=np.float32)
        return X, t_idx, a_idx, y

    Xtr, ttr, atr, ytr = featurize_rows(train_df)
    Xva, tva, ava, yva = featurize_rows(val_df)
    Xte, tte, ate, yte = featurize_rows(test_df)

    print("\n[split sizes] train/val/test rows:", len(train_df), len(val_df), len(test_df))

    model = build_model(fp_dim=args.n_bits, n_targets=len(targets_sorted), n_assay_types=2)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
    ]

    model.fit(
        {"fp": Xtr, "target_id": ttr, "assay_type": atr},
        ytr,
        validation_data=({"fp": Xva, "target_id": tva, "assay_type": ava}, yva),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=2,
        callbacks=callbacks,
    )

    # Evaluate
    yhat = model.predict({"fp": Xte, "target_id": tte, "assay_type": ate}, batch_size=args.batch, verbose=0).reshape(-1)
    overall = rmse(yte, yhat)
    print("\n[Test] overall RMSE:", overall)

    report_group_rmse(test_df, yte, yhat, ["target_chembl_id"], "Test RMSE by target")
    report_group_rmse(test_df, yte, yhat, ["assay_type"], "Test RMSE by assay_type")
    report_group_rmse(test_df, yte, yhat, ["target_chembl_id", "assay_type"], "Test RMSE by target + assay_type")

    # Save predictions
    out_pred = test_df.copy()
    out_pred["y_true"] = yte
    out_pred["y_pred"] = yhat
    out_csv = os.path.join(args.outdir, "test_predictions.csv")
    out_pred.to_csv(out_csv, index=False)
    print("\n[Saved]", out_csv)

    # Optional: rough global bit importance
    if args.dump_bits:
        top = top_bits_via_gradients(model, Xte, tte, ate, n_top=50, seed=args.seed)
        bits_df = pd.DataFrame(top, columns=["bit", "importance"])
        bits_path = os.path.join(args.outdir, "top_bits_grad_times_input.csv")
        bits_df.to_csv(bits_path, index=False)
        print("[Saved]", bits_path)


if __name__ == "__main__":
    main()
