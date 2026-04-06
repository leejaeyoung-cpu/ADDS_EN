"""
conftest.py — ADDS Test Configuration
Prevents pytest from collecting legacy/ test files that require
unavailable dependencies (GPU, medical_imaging module, etc.)
"""
import pytest

# Legacy tests require external dependencies not available in all environments:
# - GPU/CUDA runtime
# - medical_imaging module (requires full installation)
# - Live OpenAI API connection
# Do not collect legacy tests unless explicitly requested with:
#   pytest tests/legacy/ --run-legacy
collect_ignore_glob = ["legacy/*"]


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "legacy: mark test as legacy (requires special environment)"
    )
    config.addinivalue_line(
        "markers",
        "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring full system"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests marked as legacy unless --run-legacy is passed."""
    if not config.getoption("--run-legacy", default=False):
        skip_legacy = pytest.mark.skip(reason="Legacy test: use --run-legacy to run")
        for item in items:
            if "legacy" in str(item.fspath):
                item.add_marker(skip_legacy)


def pytest_addoption(parser):
    """Add --run-legacy CLI option."""
    parser.addoption(
        "--run-legacy",
        action="store_true",
        default=False,
        help="Run legacy tests (requires full installation with GPU etc.)"
    )
