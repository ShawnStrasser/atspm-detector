"""The opt-in marker: tests marked bench run only under pytest -m bench."""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "bench: benchmark smoke test, opt-in; run it with pytest -m bench")


def pytest_collection_modifyitems(config, items):
    if "bench" not in (config.getoption("markexpr") or ""):
        skip = pytest.mark.skip(reason="opt-in: pytest -m bench")
        for item in items:
            if item.get_closest_marker("bench"):
                item.add_marker(skip)
