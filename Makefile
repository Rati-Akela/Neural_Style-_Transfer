.PHONY: install run test lint gallery benchmark docker

install:
	pip install -r requirements-dev.txt

run:
	streamlit run app.py

test:
	pytest

lint:
	ruff check . && ruff format --check .

gallery:
	python scripts/make_gallery.py

benchmark:
	python scripts/benchmark.py

docker:
	docker compose up --build
