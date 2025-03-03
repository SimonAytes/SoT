from pathlib import Path

def default_path():
    """Create default configuration using package directory."""
    package_dir = Path(__file__).parent.parent
    return package_dir