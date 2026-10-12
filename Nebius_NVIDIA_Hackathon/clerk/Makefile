# Clerk developer commands. Run from the repository root.
PY ?= backend/.venv/bin/python
SHELL := /bin/bash

.PHONY: setup dev test lint e2e eval eval-live eval-ablation models live-smoke docker-build docker-run check

setup:            ## install backend + frontend dependencies
	python3 -m venv backend/.venv
	$(PY) -m pip install -q -e 'backend[dev]'
	cd frontend && npm ci

dev:              ## API on :8000 + Vite on :5173 (offline unless .env says live)
	set -a; [ -f .env ] && source .env; set +a; \
	(cd backend && ../$(PY) -m uvicorn clerk.api:create_app --factory --reload --port 8000) & \
	(cd frontend && npm run dev)

test:             ## backend unit + API tests
	cd backend && ../$(PY) -m pytest

lint:             ## ruff + mypy + TypeScript
	cd backend && ../$(PY) -m ruff check . && ../$(PY) -m ruff format --check . && ../$(PY) -m mypy clerk evals scripts
	cd frontend && npx tsc --noEmit

e2e:              ## build the UI and run the Playwright browser tests
	cd frontend && npm run build && npx playwright test

eval:             ## offline eval (pipeline check, not model quality)
	cd backend && ../$(PY) -m evals.run_eval

eval-live:        ## real eval on Nebius: the numbers for the README
	set -a; source .env; set +a; cd backend && CLERK_MODE=live ../$(PY) -m evals.run_eval

eval-ablation:    ## live ablation: Super for every step
	set -a; source .env; set +a; cd backend && CLERK_MODE=live ../$(PY) -m evals.run_eval --all-super

models:           ## list Nemotron models your key can use
	set -a; source .env; set +a; $(PY) backend/scripts/list_models.py

live-smoke:       ## one real end-to-end audit against Nebius
	set -a; source .env; set +a; CLERK_MODE=live $(PY) backend/scripts/live_smoke.py

docker-build:
	docker build -t clerk:latest .

docker-run:       ## run the image with your .env on :8000
	docker run --rm -p 8000:8000 --env-file .env clerk:latest

check: lint test e2e eval   ## everything CI runs
