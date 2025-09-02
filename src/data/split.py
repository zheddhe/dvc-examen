import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    # chemins
    raw_path = os.path.join("data", "raw", "dataset.csv")  # <-- adapte le nom du fichier si besoin
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # chargement des données
    logging.info(f"Chargement des données brutes depuis {raw_path}")
    df = pd.read_csv(raw_path)

    # séparation features / cible
    target_col = "silica_concentrate"
    if target_col not in df.columns:
        raise ValueError(f"La colonne cible '{target_col}' est introuvable dans le dataset")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # split train/test
    logging.info("Découpage train/test en cours (80%/20%)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # sauvegarde
    logging.info("Sauvegarde des fichiers dans data/processed")
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    logging.info("✅ Split terminé avec succès.")


if __name__ == "__main__":
    main()
