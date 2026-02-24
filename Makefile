.PHONY: setup run test lint clean notebooks docker-build docker-run

setup:
	@echo " Installing dependencies..."
	pip install -r requirements.txt
	pip install pytest flake8 black jupyter nbconvert

run:
	@echo " Running full pipeline..."
	python run_all.py

test:
	@echo " Running test suite..."
	pytest tests/ -v

lint:
	@echo " Linting code..."
	flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics

notebooks:
	@echo " Executing notebooks..."
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

clean:
	@echo " Cleaning up..."
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .ipynb_checkpoints
	rm -rf outputs/figures/*.png
	rm -rf outputs/tables/*.csv
	rm -rf models/*.pkl

docker-build:
	docker build -t healthcare-outcomes .

docker-run:
	docker run -p 8501:8501 healthcare-outcomes

