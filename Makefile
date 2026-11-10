# make reproduce   rerun the S&P 400 event study (fetches Wikipedia and yfinance on first run) and re-render the README
# make check       verify README event-study table matches results/ (fast, used in CI)
# make test        run the unit tests
.PHONY: setup reproduce check test

setup:
	uv sync

reproduce:
	uv run python scripts/index_event_study.py

check:
	uv run python scripts/index_event_study.py --check

test:
	MPLBACKEND=Agg uv run pytest -q
