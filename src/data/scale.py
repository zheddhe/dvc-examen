import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------------------------------------------------
# Logs
# -------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "scale.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"),
              logging.StreamHandler()]
)


def main():
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    x_train_path = os.path.join(processed_dir, "X_train.csv")
    x_test_path = os.path.join(processed_dir, "X_test.csv")
    scaler_path = os.path.join(processed_dir, "scaler.pkl")

    logging.info(f"Lecture {x_train_path} et {x_test_path}")
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    logging.info("StandardScaler fit/transform")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    out_train = os.path.join(processed_dir, "X_train_scaled.csv")
    out_test = os.path.join(processed_dir, "X_test_scaled.csv")

    logging.info("Sauvegarde X_train_scaled / X_test_scaled")
    X_train_scaled.to_csv(out_train, index=False)
    X_test_scaled.to_csv(out_test, index=False)

    logging.info(f"Sauvegarde du scaler : {scaler_path}")
    joblib.dump(scaler, scaler_path)

    logging.info("✅ Scaling terminé.")


if __name__ == "__main__":
    main()
