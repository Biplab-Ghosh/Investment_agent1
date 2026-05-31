"""
Load API keys from the external APIs folder and sync into the project .env.

Default external folder: E:\\PyCharm Projects\\APIs
Override with EXTERNAL_APIS_DIR in the environment.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Project root (parent of config/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_EXTERNAL_DIR = Path(r"E:\PyCharm Projects\APIs")

# External filename -> environment variable
_KEY_FILES: dict[str, str] = {
    "openai_api_1.txt": "OPENAI_API_KEY",
    "alpha_vantage.txt": "ALPHA_VANTAGE_API_KEY",
    "FRED_API_Key.txt": "FRED_API_KEY",
}


def _parse_key_file(content: str, env_name: str) -> str:
    text = content.strip()
    if not text:
        return ""
    prefix = f"{env_name}="
    if text.startswith(prefix):
        return text[len(prefix) :].strip().strip('"').strip("'")
    if "=" in text and text.split("=", 1)[0].strip().upper() == env_name:
        return text.split("=", 1)[1].strip().strip('"').strip("'")
    return text


def external_apis_dir() -> Path:
    return Path(os.getenv("EXTERNAL_APIS_DIR", str(_DEFAULT_EXTERNAL_DIR)))


def load_keys_from_external_dir(directory: Path | None = None) -> dict[str, str]:
    """Read key files from *directory* and return {ENV_NAME: value}."""
    directory = directory or external_apis_dir()
    loaded: dict[str, str] = {}
    if not directory.is_dir():
        return loaded

    for filename, env_name in _KEY_FILES.items():
        path = directory / filename
        if not path.is_file():
            continue
        try:
            value = _parse_key_file(path.read_text(encoding="utf-8"), env_name)
        except OSError:
            continue
        if value:
            loaded[env_name] = value
    return loaded


def apply_keys_to_environ(keys: dict[str, str], overwrite: bool = False) -> list[str]:
    """Set os.environ for each key. Returns names that were applied."""
    applied: list[str] = []
    for name, value in keys.items():
        if overwrite or not os.getenv(name):
            os.environ[name] = value
            applied.append(name)
    return applied


_PLACEHOLDERS = frozenset({
    "",
    "your_openai_api_key_here",
    "your_alpha_vantage_key_here",
    "your_fred_api_key_here",
    "your_anthropic_api_key_here",
})


def sync_project_env_file(keys: dict[str, str], env_path: Path | None = None) -> None:
    """
    Merge *keys* into the project .env without overwriting existing non-placeholder values.
    Creates .env from .env.example when missing.
    """
    env_path = env_path or (_PROJECT_ROOT / ".env")
    if env_path.is_file():
        content = env_path.read_text(encoding="utf-8")
    else:
        example = _PROJECT_ROOT / ".env.example"
        content = example.read_text(encoding="utf-8") if example.is_file() else ""

    current: dict[str, str] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k, _, v = stripped.partition("=")
            current[k.strip()] = v.strip()

    for name, value in keys.items():
        if current.get(name) and current[name] not in _PLACEHOLDERS:
            continue
        pattern = rf"^{re.escape(name)}=.*$"
        replacement = f"{name}={value}"
        if re.search(pattern, content, flags=re.MULTILINE):
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        else:
            content = content.rstrip() + f"\n{name}={value}\n"

    env_path.write_text(content, encoding="utf-8")


def load_external_api_keys(sync_env: bool = True) -> dict[str, str]:
    """
    Load keys from the external APIs directory into os.environ and optionally .env.
    Call this before reading settings constants.
    """
    keys = load_keys_from_external_dir()
    apply_keys_to_environ(keys, overwrite=False)
    if sync_env and keys:
        sync_project_env_file(keys)
    return keys


def mask_key(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"
