.PHONY: develop test coverage docs bench bench-all wheel wheel-install clean

develop:
	maturin develop --release
	pip install -e ".[dev]"

test:
	pytest tests/

coverage:
	pytest tests/ --cov=grumpy --cov-report=term-missing --cov-fail-under=100

docs:
	mkdocs build -f mkdocs.yml

bench:
	python benchmarks/benchmark_ragged_api.py
	python benchmarks/benchmark_elementwise.py

bench-all:
	@for f in benchmarks/benchmark_*.py; do \
	  echo "=== $$f ==="; \
	  python "$$f" || exit 1; \
	done

wheel:
	maturin build --release --locked --compatibility pypi -o dist

wheel-install: wheel
	pip install dist/grumpy-*.whl --force-reinstall

clean:
	rm -rf dist/ target/wheels/ site/ htmlcov/ .coverage .pytest_cache
