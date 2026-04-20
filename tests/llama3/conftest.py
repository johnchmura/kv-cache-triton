"""Minimal conftest for Llama 3 tests: reset CUDA stats and add gpu marker."""

from __future__ import annotations

import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires a real GPU with enough HBM (tens of GiB)")


@pytest.fixture(autouse=True)
def _reset_cuda_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    yield
