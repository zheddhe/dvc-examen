import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler

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


# -------------------------------------------------------------------
# Script principal
# -------------------------------------------------------------------
def main():
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    x_train_path = os.path.join(processed_dir, "X_train.csv")
    x_test_path = os.path.join(processed_dir, "X_test.csv")

    logging.info(f"Lecture {x_train_path} et {x_test_path}")
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    # Colonnes numériques uniquement pour le scaler
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    non_num_cols = [c for c in X_train.columns if c not in num_cols]

    logging.info(f"Colonnes numériques scalées ({len(num_cols)}): {num_cols}")
    if non_num_cols:
        logging.info("Colonnes non numériques laissées inchangées "
                     f"({len(non_num_cols)}): {non_num_cols}")

    # Fit/transform sur le sous-ensemble numérique
    logging.info("StandardScaler fit/transform sur colonnes numériques")
    scaler = StandardScaler()
    X_train_num_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[num_cols]),
        columns=num_cols,
        index=X_train.index
    )
    X_test_num_scaled = pd.DataFrame(
        scaler.transform(X_test[num_cols]),
        columns=num_cols,
        index=X_test.index
    )

    # Recompose avec les colonnes non-numériques, puis réordonne comme l'original
    X_train_scaled = pd.concat([X_train_num_scaled, X_train[non_num_cols]], axis=1)
    X_test_scaled = pd.concat([X_test_num_scaled, X_test[non_num_cols]], axis=1)
    X_train_scaled = X_train_scaled[X_train.columns]
    X_test_scaled = X_test_scaled[X_test.columns]

    out_train = os.path.join(processed_dir, "X_train_scaled.csv")
    out_test = os.path.join(processed_dir, "X_test_scaled.csv")

    logging.info("Sauvegarde X_train_scaled / X_test_scaled")
    X_train_scaled.to_csv(out_train, index=False)
    X_test_scaled.to_csv(out_test, index=False)

    logging.info("✅ Scaling terminé.")


if __name__ == "__main__":
    main()
