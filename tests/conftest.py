"""
Pytest configuration for the RL2 workshop tests.

Sets up import paths and provides fixtures for testing any BatteryStorageEnv
implementation.

Usage:
    uv run pytest tests/ -v                     # Test participant's implementation (default)
    uv run pytest tests/ -v --solution <name>   # Test a solution from 02_solutions/
"""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# Module aliases: allow `from workshop.envs import ...` instead of `from 01_workshop.envs import ...`
# ---------------------------------------------------------------------------

def _create_module_alias(alias: str, real_path: Path):
    """Create a module alias pointing to a real path."""
    if alias in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(
        alias,
        real_path / "__init__.py",
        submodule_search_locations=[str(real_path)],
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)


_create_module_alias("workshop", project_root / "01_workshop")
_create_module_alias("workshop.envs", project_root / "01_workshop" / "envs")
_create_module_alias("solutions", project_root / "02_solutions")


# ---------------------------------------------------------------------------
# CLI option: --solution
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--solution",
        default=None,
        help="Name of a solution file to test (e.g. 'my_solution' for battery_env_solution_my_solution.py). "
             "If omitted, tests the workshop skeleton in 01_workshop/envs/battery_env.py.",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env_cls(request):
    """Provide the BatteryStorageEnv class based on --solution flag."""
    solution = request.config.getoption("--solution")

    if solution is None:
        from workshop.envs.battery_env import BatteryStorageEnv
    else:
        module = importlib.import_module(f"solutions.battery_env_solution_{solution}")
        BatteryStorageEnv = module.BatteryStorageEnv

    return BatteryStorageEnv


@pytest.fixture
def env(env_cls):
    """Create a BatteryStorageEnv instance (uses default split='train')."""
    env = env_cls()
    yield env
    env.close()
