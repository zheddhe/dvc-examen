import os
import logging
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------------------------
# Logs
# -------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "train.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"),
              logging.StreamHandler()]
)


def main():
    processed_dir = os.path.join("data", "processed")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    X_train = pd.read_csv(os.path.join(processed_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze("columns")

    best_path = os.path.join(models_dir, "best_params.pkl")
    logging.info(f"Chargement hyperparams: {best_path}")
    best = joblib.load(best_path)

    model = RandomForestRegressor(
        **best.get("model_base_params", {}),
        **best.get("best_params", {})
    )

    logging.info("Fit du modèle")
    model.fit(X_train, y_train)  # type: ignore

    model_path = os.path.join(models_dir, "gbr_model.pkl")
    joblib.dump(model, model_path)
    logging.info(f"Modèle sauvegardé: {model_path}")

    logging.info("✅ Entraînement terminé.")


if __name__ == "__main__":
    main()
