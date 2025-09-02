SHELL := /bin/bash
.ONESHELL:
# flags de gestion du comportement sur erreur (sortie immediate et sur premiere erreur en pipe)
.SHELLFLAGS := -eu -o pipefail -c
# cible make par defaut : help
.DEFAULT_GOAL := help

env: ## Construit l'environnement de développement initial
	@echo "==> ENV: virtual env creation"
	python3 -m venv .venv

req: ## met a jour les dependances (via requirements.txt)
	source .venv/bin/activate
	@echo "==> ENV: pip update"
	python3 -m pip install --upgrade pip
	@echo "==> ENV: install the required dependencies in the virtual env"
	pip install -r requirements.txt

dvc_set: ## configure le repo dvc S3 (dagshub) dans l'env de dev
	source .venv/bin/activate
	@echo "==> DVC_INIT: remove data models from git management"
	dvc remote add origin s3://dvc -f
	dvc remote modify origin endpointurl https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.dvc
	dvc remote modify origin --local access_key_id ${DAGSHUB_KEY}
	dvc remote modify origin --local secret_access_key ${DAGSHUB_KEY}

repro: ## Lance la pipeline dvc
	source .venv/bin/activate
	@echo "==> RUN: full dvc pipeline"
	dvc repro

help: ## Affiche cette aide
	@awk 'BEGIN{FS=":.*##"; printf "\nTargets disponibles:\n\n"} /^[a-zA-Z0-9_.-]+:.*##/{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2} /^.DEFAULT_GOAL/{print ""} ' $(MAKEFILE_LIST)

clean: ## Nettoie artefacts communs (RUF)
