#!/usr/bin/env python3
"""
sar_pipeline/pipeline.py  (TWO-HEAD MODEL VERSION)

Two-head shared-encoder model:
- Head B: predicts binding pChEMBL
- Head F: predicts functional pChEMBL

Training uses sample weights so:
- a binding row only trains the B head (F head weight=0)
- a functional row only trains the F head (B head weight=0)

This usually helps because binding vs functional behave differently and mixing them
as a single regression target (even with assay_type as input) adds noise.

Pipeline stages remain the same:
  extract   -> query ChEMBL SQLite to cleaned measurements (Parquet)
  featurize -> cache Morgan fingerprints + scaffold split (Parquet/NPY/JSON)
  train     -> train two-head Keras model + report metrics
  all       -> run everything

Run:
  python sar_pipeline/pipeline.py --chembl_sqlite "/home/wselby/.data/chembl/36/chembl_36.db" --stage all

Outputs:
  sar_pipeline/data/test_predictions_<runid>.csv  (includes y_pred_B / y_pred_F)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


# -----------------------------
# Defaults: monoamine panel
# -----------------------------
DEFAULT_TARGETS = [
    "CHEMBL228",  # SERT
    "CHEMBL238",  # DAT
    "CHEMBL222",  # NET
    "CHEMBL217",  # DRD2
    "CHEMBL214",  # HTR1A
    "CHEMBL224",  # HTR2A
]

PCHEMBL_STANDARD_TYPES = ("IC50", "XC50", "EC50", "AC50", "Ki", "Kd", "Potency", "ED50")


# -----------------------------
# Utility helpers
# -----------------------------
def sha1_dict(d: dict) -> str:
    s = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:12]


def maybe_autofind_chembl_db(path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        cands = glob.glob(os.path.join(path, "**", "chembl*.db"), recursive=True)
        if not cands:
            raise FileNotFoundError(f"No chembl*.db found under directory: {path}")
        cands = sorted(cands, key=lambda p: os.path.getsize(p), reverse=True)
        return cands[0]
    raise FileNotFoundError(path)


def connect_sqlite(path: str) -> sqlite3.Connection:
    path = maybe_autofind_chembl_db(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    uri = f"file:{path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def canonicalize_smiles(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=True)


def scaffold_smiles(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    sc = MurckoScaffold.GetScaffoldForMol(m)
    if sc is None:
        return ""
    return Chem.MolToSmiles(sc, isomericSmiles=False)


def morgan_fp_array(smi: str, radius: int, n_bits: int) -> np.ndarray | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# -----------------------------
# Stage 1: Extract
# -----------------------------
def fetch_chembl_measurements(
    conn: sqlite3.Connection,
    targets: List[str],
    min_confidence: int = 7,
    organism: str = "Homo sapiens",
    single_protein_only: bool = True,
) -> pd.DataFrame:
    sql = f"""
    SELECT
        mdp.chembl_id          AS molecule_chembl_id,
        cs.canonical_smiles    AS smiles,
        td.chembl_id           AS target_chembl_id,
        td.pref_name           AS target_name,
        td.target_type         AS target_type,
        td.organism            AS organism,
        a.assay_type           AS assay_type,
        a.confidence_score     AS confidence_score,
        act.standard_type      AS standard_type,
        act.standard_relation  AS standard_relation,
        act.standard_units     AS standard_units,
        act.pchembl_value      AS pchembl_value
    FROM activities act
    JOIN assays a ON act.assay_id = a.assay_id
    JOIN target_dictionary td ON a.tid = td.tid
    LEFT JOIN molecule_hierarchy mh ON act.molregno = mh.molregno
    JOIN molecule_dictionary mdp
        ON mdp.molregno = COALESCE(mh.parent_molregno, act.molregno)
    JOIN compound_structures cs
        ON cs.molregno = mdp.molregno
    WHERE
        td.chembl_id IN ({",".join(["?"] * len(targets))})
        AND td.organism = ?
        AND a.assay_type IN ('B','F')
        AND act.pchembl_value IS NOT NULL
        AND act.standard_relation = '='
        AND act.standard_units = 'nM'
        AND act.standard_type IN ({",".join(["?"] * len(PCHEMBL_STANDARD_TYPES))})
        AND a.confidence_score >= ?
        {"AND td.target_type = 'SINGLE PROTEIN'" if single_protein_only else ""}
    ;
    """
    params: List = []
    params.extend(targets)
    params.append(organism)
    params.extend(list(PCHEMBL_STANDARD_TYPES))
    params.append(int(min_confidence))

    df = pd.read_sql(sql, conn, params=params)
    df = df.dropna(subset=["smiles", "pchembl_value", "target_chembl_id", "assay_type", "standard_type"])
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value"])
    return df


def stage_extract(args) -> str:
    out_parquet = os.path.join(args.data_dir, f"measurements_{args.run_id}.parquet")

    with connect_sqlite(args.chembl_sqlite) as conn:
        df = fetch_chembl_measurements(
            conn,
            targets=args.targets,
            min_confidence=args.min_confidence,
            single_protein_only=(not args.allow_non_single_protein),
        )

    tqdm.pandas(desc="Canonical SMILES")
    df["smiles"] = df["smiles"].progress_apply(lambda s: canonicalize_smiles(s))
    df = df.dropna(subset=["smiles"])

    # IMPORTANT: do not mix endpoints -> include standard_type
    df = df.groupby(["smiles", "target_chembl_id", "assay_type", "standard_type"], as_index=False).agg(
        pchembl_value=("pchembl_value", "median"),
        target_name=("target_name", "first"),
    )

    df.to_parquet(out_parquet, index=False)

    # QA
    print("[extract] saved:", out_parquet)
    print("[extract] rows:", len(df), "unique_smiles:", df["smiles"].nunique())
    print("\n[extract] rows by target+assay:")
    print(df.groupby(["target_chembl_id", "assay_type"]).size().sort_values(ascending=False).to_string())
    print("\n[extract] rows by target+assay+standard_type (top 30):")
    print(df.groupby(["target_chembl_id", "assay_type", "standard_type"]).size()
          .sort_values(ascending=False).head(30).to_string())
    print("\n[extract] pchembl summary by assay_type:")
    print(df.groupby("assay_type")["pchembl_value"].describe().to_string())

    return out_parquet


# -----------------------------
# Stage 2: Featurize + Split
# -----------------------------
@dataclass
class SplitIndex:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def greedy_scaffold_split(unique_smiles: List[str], frac_val=0.15, frac_test=0.15, seed=0) -> SplitIndex:
    rng = np.random.default_rng(seed)
    scaff_to_idx: Dict[str, List[int]] = {}
    for i, smi in enumerate(unique_smiles):
        sc = scaffold_smiles(smi)
        scaff_to_idx.setdefault(sc, []).append(i)

    items = list(scaff_to_idx.items())
    rng.shuffle(items)
    items.sort(key=lambda x: len(x[1]), reverse=True)

    n = len(unique_smiles)
    n_test = int(round(frac_test * n))
    n_val = int(round(frac_val * n))

    test, val, train = [], [], []
    for _, idxs in items:
        if len(test) + len(idxs) <= n_test:
            test.extend(idxs)
        elif len(val) + len(idxs) <= n_val:
            val.extend(idxs)
        else:
            train.extend(idxs)

    return SplitIndex(np.array(train, int), np.array(val, int), np.array(test, int))


def stage_featurize(args) -> Tuple[str, str, str]:
    df = pd.read_parquet(args.measurements)

    unique_smiles = df["smiles"].unique().tolist()
    split = greedy_scaffold_split(unique_smiles, seed=args.seed)

    split_map = {}
    train_set = set(split.train.tolist())
    val_set = set(split.val.tolist())
    for i, smi in enumerate(unique_smiles):
        if i in train_set:
            split_map[smi] = 0
        elif i in val_set:
            split_map[smi] = 1
        else:
            split_map[smi] = 2
    df["mol_split"] = df["smiles"].map(split_map).astype(np.int8)  # 0 train, 1 val, 2 test

    # Categorical encodings
    targets_sorted = sorted(df["target_chembl_id"].unique().tolist())
    target_to_idx = {t: i for i, t in enumerate(targets_sorted)}
    std_types_sorted = sorted(df["standard_type"].unique().tolist())
    std_to_idx = {s: i for i, s in enumerate(std_types_sorted)}

    df["t_idx"] = df["target_chembl_id"].map(target_to_idx).astype(np.int32)
    df["st_idx"] = df["standard_type"].map(std_to_idx).astype(np.int32)

    # assay_type: B->0, F->1 and also store weights for two-head training
    df["a_idx"] = df["assay_type"].map({"B": 0, "F": 1}).astype(np.int32)

    # Fingerprints cache
    fp_meta = {"radius": args.radius, "n_bits": args.n_bits, "n_unique_smiles": len(unique_smiles)}
    fp_key = sha1_dict(fp_meta)
    X_path = os.path.join(args.cache_dir, f"X_morgan_{fp_key}.npy")
    smi_path = os.path.join(args.cache_dir, f"smiles_{fp_key}.json")

    if not os.path.exists(X_path):
        print("[featurize] computing fingerprints…")
        X = np.zeros((len(unique_smiles), args.n_bits), dtype=np.float32)
        bad = 0
        for i, smi in enumerate(tqdm(unique_smiles, desc="Morgan FP")):
            arr = morgan_fp_array(smi, radius=args.radius, n_bits=args.n_bits)
            if arr is None:
                bad += 1
                continue
            X[i, :] = arr
        if bad:
            print(f"[featurize] warn: {bad} SMILES failed parsing for FP.")
        np.save(X_path, X)
        with open(smi_path, "w", encoding="utf-8") as f:
            json.dump(unique_smiles, f)
        print("[featurize] saved:", X_path)
    else:
        print("[featurize] using cached:", X_path)

    # Row-level dataset references molecule index
    smi_to_molidx = {s: i for i, s in enumerate(unique_smiles)}
    df["mol_idx"] = df["smiles"].map(smi_to_molidx).astype(np.int32)

    rows_path = os.path.join(args.data_dir, f"rows_{args.run_id}.parquet")
    df.to_parquet(rows_path, index=False)

    meta_path = os.path.join(args.data_dir, f"meta_{args.run_id}.json")
    meta = {
        "run_id": args.run_id,
        "targets_sorted": targets_sorted,
        "target_to_idx": target_to_idx,
        "std_types_sorted": std_types_sorted,
        "std_to_idx": std_to_idx,
        "n_targets": len(targets_sorted),
        "n_std_types": len(std_types_sorted),
        "fp_key": fp_key,
        "X_path": X_path,
        "smiles_path": smi_path,
        "rows_path": rows_path,
        "measurements_path": args.measurements,
        "radius": args.radius,
        "n_bits": args.n_bits,
        "seed": args.seed,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[featurize] saved rows:", rows_path)
    print("[featurize] saved meta:", meta_path)
    print("[featurize] std_types:", std_types_sorted)

    return meta_path, rows_path, X_path


# -----------------------------
# Stage 3: Train + Evaluate (two-head)
# -----------------------------
def build_two_head_model(
    fp_dim: int,
    n_targets: int,
    n_std_types: int,
    target_emb: int = 16,
    std_emb: int = 6,
) -> tf.keras.Model:
    """
    Shared encoder over:
      - Morgan FP
      - target embedding
      - standard_type embedding
    Two heads:
      - y_B (binding)
      - y_F (functional)

    We will train using sample weights to ensure each row trains only the correct head.
    """
    fp_in = tf.keras.Input(shape=(fp_dim,), name="fp")
    t_in = tf.keras.Input(shape=(), dtype="int32", name="target_id")
    st_in = tf.keras.Input(shape=(), dtype="int32", name="standard_type")

    t_emb = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(n_targets, target_emb)(t_in))
    st_emb = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(n_std_types, std_emb)(st_in))

    x = tf.keras.layers.Concatenate()([fp_in, t_emb, st_emb])
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.15)(x)

    yB = tf.keras.layers.Dense(1, name="y_B")(x)
    yF = tf.keras.layers.Dense(1, name="y_F")(x)

    model = tf.keras.Model(inputs={"fp": fp_in, "target_id": t_in, "standard_type": st_in},
                           outputs={"y_B": yB, "y_F": yF})

    # Same loss for each head; sample_weight masks per row.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss={"y_B": tf.keras.losses.Huber(delta=0.5), "y_F": tf.keras.losses.Huber(delta=0.5)},
        metrics={"y_B": [tf.keras.metrics.RootMeanSquaredError(name="rmse")],
                 "y_F": [tf.keras.metrics.RootMeanSquaredError(name="rmse")]},
    )
    return model


def make_tf_dataset_two_head(rows: pd.DataFrame, X_mol: np.ndarray, batch: int, shuffle: bool, seed: int):
    """
    Yields:
      X: dict(fp, target_id, standard_type)
      y: dict(y_B, y_F)  (both equal to pchembl_value)
      sample_weight: dict(y_B, y_F)  (mask: 1 for correct head, 0 for other)
    """
    mol_idx = rows["mol_idx"].to_numpy(np.int32)
    t_idx = rows["t_idx"].to_numpy(np.int32)
    st_idx = rows["st_idx"].to_numpy(np.int32)
    a_idx = rows["a_idx"].to_numpy(np.int32)  # 0 B, 1 F
    y = rows["pchembl_value"].to_numpy(np.float32)

    def gen():
        for mi, ti, si, ai, yi in zip(mol_idx, t_idx, st_idx, a_idx, y):
            x = {"fp": X_mol[mi], "target_id": ti, "standard_type": si}
            ydict = {"y_B": yi, "y_F": yi}
            # mask weights
            wB = 1.0 if ai == 0 else 0.0
            wF = 1.0 if ai == 1 else 0.0
            wdict = {"y_B": wB, "y_F": wF}
            yield x, ydict, wdict

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                "fp": tf.TensorSpec(shape=(X_mol.shape[1],), dtype=tf.float32),
                "target_id": tf.TensorSpec(shape=(), dtype=tf.int32),
                "standard_type": tf.TensorSpec(shape=(), dtype=tf.int32),
            },
            {
                "y_B": tf.TensorSpec(shape=(), dtype=tf.float32),
                "y_F": tf.TensorSpec(shape=(), dtype=tf.float32),
            },
            {
                "y_B": tf.TensorSpec(shape=(), dtype=tf.float32),
                "y_F": tf.TensorSpec(shape=(), dtype=tf.float32),
            },
        ),
    )

    if shuffle:
        ds = ds.shuffle(50000, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


def stage_train(args) -> str:
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    rows = pd.read_parquet(meta["rows_path"])
    X_mol = np.load(meta["X_path"])

    train_rows = rows[rows["mol_split"] == 0].reset_index(drop=True)
    val_rows = rows[rows["mol_split"] == 1].reset_index(drop=True)
    test_rows = rows[rows["mol_split"] == 2].reset_index(drop=True)

    print("[train] rows train/val/test:", len(train_rows), len(val_rows), len(test_rows))
    print("[train] assay_type counts (train):\n", train_rows["assay_type"].value_counts().to_string())

    ds_tr = make_tf_dataset_two_head(train_rows, X_mol, batch=args.batch, shuffle=True, seed=args.seed)
    ds_va = make_tf_dataset_two_head(val_rows, X_mol, batch=args.batch, shuffle=False, seed=args.seed)
    ds_te = make_tf_dataset_two_head(test_rows, X_mol, batch=args.batch, shuffle=False, seed=args.seed)

    model = build_two_head_model(
        fp_dim=X_mol.shape[1],
        n_targets=meta["n_targets"],
        n_std_types=meta["n_std_types"],
    )
    model.summary()

    run_dir = os.path.join(args.models_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.logs_dir, args.run_id)),
    ]

    model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, callbacks=callbacks, verbose=2)

    # Predict on test (get both heads)
    preds = model.predict(ds_te, verbose=0)
    # preds is dict: {"y_B": (N,1), "y_F": (N,1)}
    y_pred_B = preds["y_B"].reshape(-1).astype(np.float32)
    y_pred_F = preds["y_F"].reshape(-1).astype(np.float32)

    y_true = test_rows["pchembl_value"].to_numpy(np.float32)
    assay = test_rows["assay_type"].to_numpy()

    # Select head per row
    y_pred = np.where(assay == "B", y_pred_B, y_pred_F)

    out = test_rows[["smiles", "target_chembl_id", "assay_type", "standard_type", "pchembl_value", "target_name"]].copy()
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out["y_pred_B"] = y_pred_B
    out["y_pred_F"] = y_pred_F
    out["resid"] = out["y_true"] - out["y_pred"]
    out["abs_err"] = np.abs(out["resid"])

    out_path = os.path.join(args.data_dir, f"test_predictions_{args.run_id}.csv")
    out.to_csv(out_path, index=False)
    print("[train] saved predictions:", out_path)

    # Metrics
    print("\n[Test] overall RMSE:", rmse(out["y_true"].to_numpy(), out["y_pred"].to_numpy()))

    def group_rmse(g):
        return rmse(g["y_true"].to_numpy(), g["y_pred"].to_numpy())

    by_a = out.groupby("assay_type").apply(group_rmse).sort_values()
    by_t = out.groupby("target_chembl_id").apply(group_rmse).sort_values()
    by_ta = out.groupby(["target_chembl_id", "assay_type"]).apply(group_rmse).sort_values()
    by_st = out.groupby("standard_type").apply(group_rmse).sort_values()

    print("\n[Test] RMSE by assay_type:\n", by_a.to_string())
    print("\n[Test] RMSE by target:\n", by_t.to_string())
    print("\n[Test] RMSE by target+assay:\n", by_ta.to_string())
    print("\n[Test] RMSE by standard_type:\n", by_st.to_string())

    # Also report head-specific RMSE on their own subsets
    maskB = (out["assay_type"] == "B").to_numpy()
    maskF = (out["assay_type"] == "F").to_numpy()
    if maskB.any():
        print("\n[Test] Binding head RMSE on binding rows:", rmse(out.loc[maskB, "y_true"].to_numpy(),
                                                              out.loc[maskB, "y_pred_B"].to_numpy()))
    if maskF.any():
        print("\n[Test] Functional head RMSE on functional rows:", rmse(out.loc[maskF, "y_true"].to_numpy(),
                                                                      out.loc[maskF, "y_pred_F"].to_numpy()))
    return out_path


# -----------------------------
# Main CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--chembl_sqlite", type=str, required=True, help="Path to ChEMBL SQLite .db OR a directory containing it")
    ap.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    ap.add_argument("--min_confidence", type=int, default=7)
    ap.add_argument("--allow_non_single_protein", action="store_true")

    ap.add_argument("--data_dir", default="sar_pipeline/data")
    ap.add_argument("--cache_dir", default="sar_pipeline/cache")
    ap.add_argument("--models_dir", default="sar_pipeline/models")
    ap.add_argument("--logs_dir", default="sar_pipeline/logs")

    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--n_bits", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)

    ap.add_argument("--stage", choices=["extract", "featurize", "train", "all"], default="all")
    ap.add_argument("--measurements", default="")
    ap.add_argument("--meta", default="")

    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    run_spec = {
        "targets": args.targets,
        "min_confidence": args.min_confidence,
        "allow_non_single_protein": args.allow_non_single_protein,
        "radius": args.radius,
        "n_bits": args.n_bits,
        "seed": args.seed,
        "model": "two_head_BF",
    }
    args.run_id = sha1_dict(run_spec)
    print("[run_id]", args.run_id)

    if args.stage in ("extract", "all"):
        args.measurements = stage_extract(args)

    if args.stage in ("featurize", "all"):
        if not args.measurements:
            raise SystemExit("Need --measurements for stage=featurize (or use --stage all).")
        meta_path, _, _ = stage_featurize(args)
        args.meta = meta_path

    if args.stage in ("train", "all"):
        if not args.meta:
            raise SystemExit("Need --meta for stage=train (or use --stage all).")
        stage_train(args)


if __name__ == "__main__":
    main()
