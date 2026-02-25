# Makefile — Googly repo root
#
# Requires: GNU Make (winget install GnuWin32.Make)
# All targets run from D:\googly (repo root).
#
# Usage:
#   make api          FastAPI only, hot reload on :8000
#   make dev          docker-compose (db + redis + worker) + api
#   make worker       ARQ worker only (inside Docker)
#   make streamlit    Streamlit UI on :8501
#
#   make migrate      run pending Alembic migrations
#   make migrate-new  autogenerate new migration (set MSG= on CLI)
#   make db-reset     drop + recreate local DB (destructive)
#
#   make test         all tests
#   make test-unit    unit tests only (no docker needed)
#   make lint         ruff check
#   make format       ruff format
#   make typecheck    mypy
#   make check-all    lint + typecheck + test-unit
#
#   make check-env    verify required env vars are present
#   make kill-api     kill any process holding :8000

SHELL := cmd.exe

# Paths
API_DIR      := apps\api
# These paths are used AFTER cd $(API_DIR), so they are relative to apps\api
VENV_PYTHON  := .venv\Scripts\python.exe
VENV_UVICORN := .venv\Scripts\uvicorn.exe

# Port the API runs on
API_PORT := 8000

# ─── helpers ────────────────────────────────────────────────────────────────

# Kill whatever owns API_PORT (idempotent — succeeds even if port is free).
# Uses PowerShell because pure cmd has no netstat-kill one-liner.
.PHONY: kill-api
kill-api:
	@powershell -NoProfile -Command " \
		$$conns = Get-NetTCPConnection -LocalPort $(API_PORT) -ErrorAction SilentlyContinue; \
		$$pids = $$conns | Select-Object -ExpandProperty OwningProcess -Unique; \
		if ($$pids) { \
			foreach ($$p in $$pids) { \
				$$proc = Get-WmiObject Win32_Process -Filter \"ProcessId=$$p\" -ErrorAction SilentlyContinue; \
				if ($$proc) { \
					Write-Host \"Killing PID $$p ($$($proc.Name)) on port $(API_PORT)\"; \
					Stop-Process -Id $$p -Force -ErrorAction SilentlyContinue; \
				} else { \
					Write-Host \"PID $$p is a WSL/Docker ghost on port $(API_PORT) — restarting WSL network\"; \
					wsl --shutdown 2>$$null; \
					Start-Sleep -Seconds 2; \
				} \
			} \
			Start-Sleep -Milliseconds 800; \
		} else { \
			Write-Host 'Port $(API_PORT) is free'; \
		} \
	"

# ─── primary targets ────────────────────────────────────────────────────────

# Start FastAPI with hot reload. Kills any stale process on :8000 first.
.PHONY: api
api: kill-api
	@echo Starting API on port $(API_PORT) with hot reload...
	cd $(API_DIR) && $(VENV_PYTHON) -m uvicorn app.main:app \
		--host 0.0.0.0 \
		--port $(API_PORT) \
		--reload \
		--reload-dir app

# Start all local services (db + redis + worker) then the API.
.PHONY: dev
dev:
	@echo Starting docker services...
	docker compose up -d db redis worker
	@echo Waiting for services to be healthy...
	@powershell -NoProfile -Command "Start-Sleep 5"
	$(MAKE) api

# ARQ worker (runs inside Docker via docker-compose).
.PHONY: worker
worker:
	docker compose up worker

# Streamlit UI.
.PHONY: streamlit
streamlit:
	cd apps\streamlit && ..\..\$(VENV_PYTHON) -m streamlit run app.py --server.port 8501

# ─── database ───────────────────────────────────────────────────────────────

.PHONY: migrate
migrate:
	cd $(API_DIR) && $(VENV_PYTHON) -m alembic upgrade head

.PHONY: migrate-new
migrate-new:
	@if "$(MSG)"=="" ( echo "Usage: make migrate-new MSG=your_description" & exit 1 )
	cd $(API_DIR) && $(VENV_PYTHON) -m alembic revision --autogenerate -m "$(MSG)"

.PHONY: db-reset
db-reset:
	@echo WARNING: This will DROP and recreate the local database.
	@powershell -NoProfile -Command "Read-Host 'Press Enter to continue, Ctrl+C to abort'" 
	cd $(API_DIR) && $(VENV_PYTHON) -m alembic downgrade base
	cd $(API_DIR) && $(VENV_PYTHON) -m alembic upgrade head

# ─── quality gates ──────────────────────────────────────────────────────────

.PHONY: test
test:
	cd $(API_DIR) && $(VENV_PYTHON) -m pytest tests\ -v

.PHONY: test-unit
test-unit:
	cd $(API_DIR) && $(VENV_PYTHON) -m pytest tests\unit\ -v -m "not integration"

.PHONY: lint
lint:
	cd $(API_DIR) && $(VENV_PYTHON) -m ruff check app

.PHONY: format
format:
	cd $(API_DIR) && $(VENV_PYTHON) -m ruff format app

.PHONY: typecheck
typecheck:
	cd $(API_DIR) && $(VENV_PYTHON) -m mypy app

.PHONY: check-all
check-all: lint typecheck test-unit

# ─── environment check ──────────────────────────────────────────────────────

.PHONY: check-env
check-env:
	@powershell -NoProfile -Command " \
		$$env_file = '.env'; \
		$$required = @('OPENAI_API_KEY','DATABASE_URL','REDIS_URL','APIM_INTERNAL_TOKEN','CLERK_SECRET_KEY'); \
		if (-not (Test-Path $$env_file)) { Write-Error '.env file not found'; exit 1 } \
		$$content = Get-Content $$env_file; \
		$$missing = @(); \
		foreach ($$key in $$required) { \
			if (-not ($$content | Select-String -Pattern \"^$$key=\")) { \
				$$missing += $$key; \
			} \
		} \
		if ($$missing.Count -gt 0) { \
			Write-Error \"Missing required env vars: $$($$missing -join ', ')\"; exit 1 \
		} else { \
			Write-Host 'All required env vars present' \
		} \
	"
