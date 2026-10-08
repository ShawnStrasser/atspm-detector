"""pytest entry point: the package's own check must pass (same as the atspm-detector-check command)."""
from atspm_detector.check import main


def test_package_check():
    assert main([]) == 0
