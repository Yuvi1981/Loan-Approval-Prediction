.PHONY: setup train app drift-demo fairness test clean lint

PYTHON = .venv/Scripts/python
STREAMLIT = .venv/Scripts/streamlit
PYTEST = .venv/Scripts/pytest

setup:
	@echo "Setting up dependencies..."
	pip install -r requirements.txt

train:
	@echo "Training models and logging to MLflow..."
	$(PYTHON) -m src.train

app:
	@echo "Launching Streamlit dashboard..."
	$(STREAMLIT) run app/streamlit_app.py

drift-demo:
	@echo "Running drift detection simulation..."
	$(PYTHON) -m src.simulate_drift

fairness:
	@echo "Running fairness audit..."
	$(PYTHON) -m src.fairness

test:
	@echo "Running test suite..."
	$(PYTEST) tests/ -v

lint:
	@echo "Running lint check..."
	$(PYTHON) -m ruff check src tests app

clean:
	@echo "Cleaning temporary files..."
	rm -rf __pycache__ .pytest_cache
