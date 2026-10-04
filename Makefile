.PHONY: install test lint format eval run chat docker

install:
	pip install -e ".[dev]"

test:
	pytest --cov --cov-report=term-missing

lint:
	ruff check src tests && ruff format --check src tests

format:
	ruff format src tests && ruff check --fix src tests

eval:
	python -m support_agent.evaluation --min-accuracy 0.90 --out eval/report.json

run:
	uvicorn support_agent.api:app_factory --factory --reload --port 8000

chat:
	python -m support_agent.cli

docker:
	docker compose up --build
