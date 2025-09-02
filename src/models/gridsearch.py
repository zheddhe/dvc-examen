import os
import logging
import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# -------------------------------------------------------------------
# Logs
# -------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "gridsearch.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"),
              logging.StreamHandler()]
)


def load_params(params_path: str = "params.yaml") -> dict:
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    # fallback minimal si pas de params.yaml
    return {
        "model": {"random_state": 42},
        "search": {
            "cv": 5, "n_jobs": -1,
            "param_grid": {
                "n_estimators": [100, 300],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            }
        }
    }


def main():
    processed_dir = os.path.join("data", "processed")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    X_train = pd.read_csv(os.path.join(processed_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze("columns")

    params = load_params("params.yaml")
    model_base = params.get("model", {})
    search_cfg = params.get("search", {})
    param_grid = search_cfg.get("param_grid", {})
    cv = search_cfg.get("cv", 5)
    n_jobs = search_cfg.get("n_jobs", -1)

    logging.info(f"Base model params: {model_base}")
    logging.info(f"Grid: {param_grid} | CV={cv} | n_jobs={n_jobs}")

    base_model = RandomForestRegressor(**model_base)
    gs = GridSearchCV(
        base_model, param_grid=param_grid, cv=cv,
        n_jobs=n_jobs, scoring="neg_mean_squared_error"
    )
    gs.fit(X_train, y_train)  # type: ignore

    best = {
        "best_params": gs.best_params_,
        "best_score": float(gs.best_score_),
        "cv": cv,
        "model_class": "RandomForestRegressor",
        "model_base_params": model_base
    }
    best_pkl = os.path.join(models_dir, "best_params.pkl")
    joblib.dump(best, best_pkl)
    logging.info(f"Best params sauvegardés: {best_pkl}")

    logging.info("✅ GridSearch terminé.")


if __name__ == "__main__":
    main()
