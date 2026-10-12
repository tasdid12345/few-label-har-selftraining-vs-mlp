"""Baseline (supervised MLP, 5 labels/class) vs improved (LightGBM confidence self-training) on OpenML HAR."""
import json, os, sys, tempfile, time
import numpy as np, pandas as pd, requests, lightgbm as lgb
import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
SEEDS = [42, 43, 44]          # both models run with each seed
DATA_ID = 1478                # OpenML "har" (561 features, 6 activities)
SHOTS = 5                     # labeled rows per class
THRESH = 0.9                  # pseudo-label confidence threshold
ROUNDS = 3                    # self-training iterations
META_URL = f"https://www.openml.org/api/v1/json/data/{DATA_ID}"


def get(url, tries=5):
    """HTTP GET with timeout=60 and exponential backoff on 429/5xx."""
    for i in range(tries):
        r = requests.get(url, timeout=60)
        if r.status_code in (429, 500, 502, 503, 504) and i < tries - 1:
            print(f"HTTP {r.status_code}, retrying in {2 ** i}s")
            time.sleep(2 ** i)
            continue
        r.raise_for_status()
        return r


def load_har():
    """Download the OpenML ARFF file and return features, string labels, name, url."""
    desc = get(META_URL).json()["data_set_description"]
    print(f"Downloading {desc['name']} ARFF from OpenML...")
    raw = get(desc["url"]).content
    with tempfile.NamedTemporaryFile(suffix=".arff", delete=False) as f:
        f.write(raw)
        path = f.name
    data, _ = arff.loadarff(path)
    os.remove(path)
    df = pd.DataFrame(data)
    target = desc["default_target_attribute"]
    y = df.pop(target).map(lambda v: v.decode() if isinstance(v, bytes) else str(v)).to_numpy()
    return df.astype(float).to_numpy(), y, desc["name"], desc["url"]


def sample_shots(y_pool, seed):
    """Pick SHOTS labeled pool rows per class; the rest of the pool is unlabeled."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.where(y_pool == c)[0], SHOTS, replace=False)
                           for c in np.unique(y_pool)])


def run_baseline(Xl, yl, Xt, seed):
    """Supervised MLP trained only on the labeled shots (fixed, untuned hyper-parameters)."""
    m = MLPClassifier(hidden_layer_sizes=(128,), max_iter=2000, random_state=seed)
    m.fit(Xl, yl)
    return m.predict(Xt)


def run_selftrain(Xl, yl, Xu, Xt, seed):
    """LightGBM self-training: each round, add unlabeled rows with max prob > THRESH
    as pseudo-labelled data and refit from scratch on labeled + pseudo-labelled rows."""
    def fit(X, y):
        return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                  min_child_samples=2, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.5, random_state=seed, n_jobs=2,
                                  verbose=-1).fit(X, y)
    model = fit(Xl, yl)
    n_pseudo = 0
    for r in range(ROUNDS):
        proba = model.predict_proba(Xu)
        keep = proba.max(1) > THRESH
        n_pseudo = int(keep.sum())
        print(f"  round {r + 1}: {n_pseudo} confident pseudo-labels from {len(Xu)} unlabeled")
        X = np.vstack([Xl, Xu[keep]])
        y = np.concatenate([yl, model.classes_[proba[keep].argmax(1)]])
        model = fit(X, y)
    return model.predict(Xt), n_pseudo


def main():
    X, y_raw, name, url = load_har()
    classes, y = np.unique(y_raw, return_inverse=True)  # integer-encode labels
    print(f"Loaded {X.shape[0]} rows x {X.shape[1]} features, {len(classes)} classes")

    # One stratified 80/20 split with fixed seed; test set is only used for final scoring
    Xp, Xte, yp, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    sc = StandardScaler().fit(Xp)  # scaler fit on pool only (no test leakage)
    Xp, Xte = sc.transform(Xp), sc.transform(Xte)

    b_f1, b_acc, i_f1, i_acc, n_ps = [], [], [], [], []
    for s in SEEDS:
        print(f"Seed {s}")
        li = sample_shots(yp, s)
        ui = np.setdiff1d(np.arange(len(yp)), li)
        pb = run_baseline(Xp[li], yp[li], Xte, s)
        pi, npl = run_selftrain(Xp[li], yp[li], Xp[ui], Xte, s)
        b_f1.append(f1_score(yte, pb, average="macro")); b_acc.append(accuracy_score(yte, pb))
        i_f1.append(f1_score(yte, pi, average="macro")); i_acc.append(accuracy_score(yte, pi))
        n_ps.append(npl)
        print(f"  baseline macro-F1 {b_f1[-1]:.4f} | improved macro-F1 {i_f1[-1]:.4f}")

    scores = np.array([b_f1, i_f1])
    if not np.all(np.isfinite(scores)):
        print("Non-finite scores; cannot produce meaningful metrics")
        sys.exit(1)

    bm, im = float(np.mean(b_f1)), float(np.mean(i_f1))
    bs, is_ = float(np.std(b_f1)), float(np.std(i_f1))
    results = {
        "question": "Does confidence-based self-training with LightGBM, using unlabeled target-domain "
                    "windows, beat a supervised MLP trained on the same few labels for HAR with 5 labels per activity?",
        "method": "Stratified 80/20 pool/test split (seed 42). 5 labeled rows per activity from the pool; "
                  "remaining pool rows are unlabeled. Baseline: MLP on the shots. Improved: LightGBM self-trained "
                  "with pseudo-labels (p>0.9, 3 rounds). Macro-F1 on the untouched test set, mean over seeds 42-44.",
        "dataset": {
            "name": name, "source_url": url, "n_rows": int(X.shape[0]),
            "n_train": int(len(yp)), "n_test": int(len(yte)), "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro_f1",
        "higher_is_better": True,
        "baseline": "Supervised scikit-learn MLPClassifier (one hidden layer of 128, default Adam) "
                    "trained only on the 5-shot labeled rows; stand-in for the paper's supervised deep HAR model.",
        "improved": "LightGBM classifier self-trained on the 5-shot labels plus confident pseudo-labels "
                    "(max probability > 0.9) for unlabeled pool rows, iterated for 3 rounds.",
        "metrics": {
            "baseline_score": bm,
            "improved_score": im,
            "improvement": im - bm,
            "baseline_score_std": bs,
            "improved_score_std": is_,
            "baseline_f1": bm,
            "improved_f1": im,
            "baseline_accuracy": float(np.mean(b_acc)),
            "improved_accuracy": float(np.mean(i_acc)),
            "mean_pseudo_labels_final_round": float(np.mean(n_ps)),
        },
        "notes": "No hyper-parameter search was run: all settings are fixed a priori, so nothing was tuned "
                 "on the test set (and no CV tuning was needed). Only 5 labels per class, so results vary with "
                 "the shot sample; the std is over 3 seeds (population std). Scaler is fit on the pool, "
                 "including unlabeled rows (no test leakage). The baseline is a stand-in, not the paper's "
                 "NLP-label transfer method. Dataset is small (about 10k rows), so no subsampling was needed.",
    }

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.bar(["baseline MLP", "improved LightGBM ST"], [bm, im], yerr=[bs, is_], capsize=6,
            color=["grey", "steelblue"])
    plt.ylabel("macro-F1 (test)"); plt.title("HAR, 5 labels per class (mean +/- std, 3 seeds)")
    plt.tight_layout(); plt.savefig("figures/macro_f1_comparison.png", dpi=120); plt.close()

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Baseline macro-F1 {bm:.4f} +/- {bs:.4f} | Improved {im:.4f} +/- {is_:.4f}")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
