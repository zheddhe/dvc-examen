import os
import json
import logging
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------------------------
# Logs
# -------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "eval.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"),
              logging.StreamHandler()]
)


def main():
    processed_dir = os.path.join("data", "processed")
    metrics_dir = "metrics"
    models_dir = "models"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "gbr_model.pkl")
    logging.info(f"Chargement modèle: {model_path}")
    model = joblib.load(model_path)

    X_test = pd.read_csv(os.path.join(processed_dir, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze("columns")

    logging.info("Prédictions")
    y_pred = model.predict(X_test)

    # Sauvegarde des prédictions
    pred_path = os.path.join(processed_dir, "predictions.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(pred_path, index=False)
    logging.info(f"Prédictions sauvegardées: {pred_path}")

    # Métriques
    mse = float(mean_squared_error(y_test, y_pred))  # type: ignore
    mae = float(mean_absolute_error(y_test, y_pred))  # type: ignore
    r2 = float(r2_score(y_test, y_pred))  # type: ignore
    metrics = {"mse": mse, "mae": mae, "r2": r2}

    metrics_path = os.path.join(metrics_dir, "scores.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Métriques sauvegardées: {metrics_path} | {metrics}")

    logging.info("✅ Évaluation terminée.")


if __name__ == "__main__":
    main()
