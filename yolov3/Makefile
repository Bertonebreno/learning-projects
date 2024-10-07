init:
	@poetry install

all:
	@poetry run mypy .
	@poetry run black .
	@poetry run isort .
	@poetry run flake8 .

run:
	@poetry run python main.py

check_results:
	@poetry run python yolo/check_results.py
