.PHONY: reproduce render check test

reproduce:  ## run the study from data/snapshot/ and refresh the README
	uv run python scripts/reproduce.py
	uv run python scripts/render_readme.py

render:
	uv run python scripts/render_readme.py

check:
	uv run python scripts/render_readme.py --check

test:
	uv run python -m pytest -q
