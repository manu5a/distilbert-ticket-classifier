.PHONY: install data train test lint run up down

install:          ## install CPU PyTorch + all dependencies
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements-dev.txt

data:             ## regenerate the sample dataset
	python scripts/generate_sample_data.py

train:            ## fine-tune DistilBERT on data/sample_tickets.csv
	python -m ml.train

test:             ## run the test suite with coverage
	pytest --cov=app --cov=ml --cov-report=term-missing

lint:
	ruff check .

run:              ## run the API locally (needs PostgreSQL, see `make up`)
	uvicorn app.main:app --reload

up:               ## start PostgreSQL + API in Docker
	docker compose up --build

down:
	docker compose down
