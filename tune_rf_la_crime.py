from __future__ import annotations

# Randomized search for RandomForest n_estimators and max_depth (same prep as train_la_crime).
# CV picks hyperparameters (no peeking at test). Final line prints held-out test metrics once.
#
#   .venv/bin/python tune_rf_la_crime.py                      # default: CV maximizes accuracy
#   .venv/bin/python tune_rf_la_crime.py --scoring f1_macro   # CV maximizes macro-F1 instead

import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV

from train_la_crime import DEFAULT_DATA, build_rf_pipeline, make_train_test_with_collapse


def _tune_one_target(
    data_path: Path,
    max_rows: int | None,
    target: str,
    *,
    top_k: int,
    test_size: float,
    random_state: int,
    premise_top_n: int,
    weapon_top_n: int,
    district_top_n: int,
    cv: int,
    n_iter: int,
    scoring: str,
) -> None:
    X_train, X_test, y_train, y_test, _, _ = make_train_test_with_collapse(
        data_path,
        max_rows,
        top_k,
        target,
        test_size,
        random_state,
        premise_top_n,
        weapon_top_n,
        district_top_n,
    )
    n = len(y_train) + len(y_test)

    pipe = build_rf_pipeline(random_state=random_state, n_estimators=100, max_depth=20)
    param_distributions = {
        "clf__n_estimators": [50, 80, 100, 120, 150, 200, 250, 300, 400],
        "clf__max_depth": [None, 8, 12, 16, 20, 24, 28, 32, 40],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)

    cv_metric = "accuracy" if scoring == "accuracy" else "macro-F1"
    print(f"\n========== target={target}  rows={n:,} ==========")
    print(
        f"CV best {cv_metric} (mean across folds, used to pick params): {search.best_score_:.4f}"
    )
    print(f"Best clf__n_estimators: {search.best_params_['clf__n_estimators']}")
    md = search.best_params_["clf__max_depth"]
    print(f"Best clf__max_depth: {md if md is not None else 'None'}")
    print(f"Held-out test  accuracy: {acc:.4f}")
    print(f"Held-out test  macro-F1: {f1m:.4f}")
    suf = "coarse4" if target == "coarse4" else "eight"
    md_cli = 0 if md is None else md
    print(
        "Use with compare script e.g.: "
        f"--rf-n-estimators-{suf} {search.best_params_['clf__n_estimators']} "
        f"--rf-max-depth-{suf} {md_cli}"
        + ("  (0 = unlimited depth)" if md is None else "")
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tune RF n_estimators and max_depth via randomized CV on train split."
    )
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    p.add_argument(
        "--max-rows",
        type=int,
        default=80_000,
        help="Rows read from CSV; lower = faster (try 50k–120k). 0 = no cap.",
    )
    p.add_argument(
        "--target",
        choices=("coarse4", "eight", "both"),
        default="coarse4",
    )
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--premise-top-n", type=int, default=30)
    p.add_argument("--weapon-top-n", type=int, default=25)
    p.add_argument("--district-top-n", type=int, default=100)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument(
        "--n-iter",
        type=int,
        default=24,
        help="Random search trials (samples param pairs with replacement).",
    )
    p.add_argument(
        "--scoring",
        choices=("accuracy", "f1_macro"),
        default="accuracy",
        help="Metric maximized during CV on the train fold(s). Test set is only evaluated at the end.",
    )
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    max_rows = None if args.max_rows == 0 else args.max_rows
    kw = dict(
        top_k=args.top_k,
        test_size=args.test_size,
        random_state=args.random_state,
        premise_top_n=args.premise_top_n,
        weapon_top_n=args.weapon_top_n,
        district_top_n=args.district_top_n,
        cv=args.cv,
        n_iter=args.n_iter,
        scoring=args.scoring,
    )

    targets = ["coarse4", "eight"] if args.target == "both" else [args.target]
    for t in targets:
        _tune_one_target(data_path, max_rows, t, **kw)


if __name__ == "__main__":
    main()
