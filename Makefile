PYTHON ?= python
INDEXMEM_PATHS = kvpress/indexmem tests/indexmem scripts evaluation/evaluate_indexmem.py evaluation/evaluate_indexmem_sharded.py

.PHONY: test test-kvpress style format

test:
	$(PYTHON) -m pytest tests/indexmem tests/evaluation

test-kvpress:
	$(PYTHON) -m pytest tests/presses

style:
	$(PYTHON) -m black --check $(INDEXMEM_PATHS)
	$(PYTHON) -m isort --check-only $(INDEXMEM_PATHS)

format:
	$(PYTHON) -m isort $(INDEXMEM_PATHS)
	$(PYTHON) -m black $(INDEXMEM_PATHS)
