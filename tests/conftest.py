"""
Pytest configuration for the RL2 workshop tests.

Sets up import paths so tests can import from workshop and solutions packages.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Create module aliases for cleaner imports
# This allows: `from workshop.envs import BatteryStorageEnv`
# instead of: `from 01_workshop.envs import BatteryStorageEnv`
import importlib.util


def _create_module_alias(alias: str, real_path: Path):
    """Create a module alias pointing to a real path."""
    if alias in sys.modules:
        return

    # Create a simple module that re-exports from the real location
    spec = importlib.util.spec_from_file_location(
        alias,
        real_path / "__init__.py",
        submodule_search_locations=[str(real_path)],
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)


# Set up aliases
_create_module_alias("workshop", project_root / "01_workshop")
_create_module_alias("workshop.envs", project_root / "01_workshop" / "envs")
_create_module_alias("solutions", project_root / "02_solutions")
