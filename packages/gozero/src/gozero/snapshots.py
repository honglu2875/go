"""Content-addressed source/config snapshots, independent of Git availability.

These are tamper-evident source bundles, not hermetic runtime containers.
Dependency locks are captured; execution environment qualification is separate.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any


SOURCE_ROOTS = ("packages", "crates", "production", "eval", "ops", "tests")
ROOT_FILES = (
    "pyproject.toml", "uv.lock", ".python-version", "Cargo.toml", "Cargo.lock",
    "rust-toolchain.toml", "README.md", "DESIGN.md", "MULTISTEP_SEARCH.md",
)
EXCLUDED_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "target", "build", "dist"}
SOURCE_SUFFIXES = {".py", ".rs", ".toml", ".lock", ".md", ".json", ".yaml", ".yml", ".cfg", ".sgf", ".sh", ".c", ".cc", ".cpp", ".h", ".hpp"}
RECIPE_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


class SnapshotError(ValueError):
    """A source bundle does not meet the snapshot contract."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise SnapshotError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    def invalid(value: str) -> None:
        raise SnapshotError(f"Nonfinite JSON constant: {value}")
    return json.loads(path.read_text(), object_pairs_hook=_pairs, parse_constant=invalid)


def _source_files(root: Path) -> list[Path]:
    if root.is_symlink():
        raise SnapshotError(f"Source symlinks are unsupported: {root}")
    result = []
    if not root.exists():
        return result
    for directory, names, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        names[:] = sorted(name for name in names if name not in EXCLUDED_DIRS and not name.startswith("."))
        for name in names:
            if (parent / name).is_symlink():
                raise SnapshotError(f"Source symlinks are unsupported: {parent / name}")
        for name in sorted(filenames):
            path = parent / name
            if name.startswith(".") or path.suffix not in SOURCE_SUFFIXES:
                continue
            if path.is_symlink() or not path.is_file():
                raise SnapshotError(f"Expected a regular source file: {path}")
            result.append(path)
    return result


def _recipe(repo: Path, recipe: Path) -> Path:
    # Reject symlinks before resolving, including linked parent directories.
    candidate = recipe if recipe.is_absolute() else repo / recipe
    for part in (candidate, *candidate.parents):
        if part.is_symlink():
            raise SnapshotError(f"Recipe symlinks are unsupported: {part}")
    candidate = candidate.resolve()
    if candidate.parent != repo / "research" / "recipes" or not candidate.is_dir():
        raise SnapshotError("Recipe must be a direct directory under research/recipes")
    if not RECIPE_NAME.fullmatch(candidate.name):
        raise SnapshotError("Recipe names use lowercase letters, digits, and underscores")
    if not (candidate / "train.py").is_file() or not (candidate / "recipe.json").is_file():
        raise SnapshotError("Recipe requires train.py and recipe.json")
    return candidate


def freeze(repo: Path, recipe: Path, config: Path, store: Path) -> Path:
    repo = repo.resolve()
    recipe = _recipe(repo, recipe)
    store = store.resolve()
    if not (repo / "uv.lock").is_file():
        raise SnapshotError("Create uv.lock before freezing a recipe")
    roots = [repo / name for name in SOURCE_ROOTS] + [recipe]
    if any(store.is_relative_to(root) for root in roots):
        raise SnapshotError("Snapshot storage must be outside source roots")
    paths = [path for root in roots for path in _source_files(root)]
    for name in ROOT_FILES:
        path = repo / name
        if path.is_symlink():
            raise SnapshotError(f"Root-file symlinks are unsupported: {path}")
        if path.is_file():
            paths.append(path)
    if any(path.suffix == ".rs" for path in paths) and not (repo / "Cargo.lock").is_file():
        raise SnapshotError("Rust sources require Cargo.lock")
    config_data = read_json(config)
    if not isinstance(config_data, dict):
        raise SnapshotError("Resolved configuration must be a JSON object")

    # Read once: the digest describes the exact bytes copied, even if a source
    # changes while snapshotting. Avoid hashing a file and copying it later.
    payloads = {path.relative_to(repo).as_posix(): (path.read_bytes(), bool(path.stat().st_mode & 0o111)) for path in paths}
    payloads["resolved_config.json"] = (canonical_json(config_data), False)
    records = {name: {"sha256": digest(data), "bytes": len(data), "executable": executable}
               for name, (data, executable) in sorted(payloads.items())}
    contract = {"schema_version": 1, "recipe": recipe.relative_to(repo).as_posix(), "files": records}
    snapshot_id = digest(canonical_json(contract))
    manifest = {**contract, "snapshot_id": snapshot_id}
    store.mkdir(parents=True, exist_ok=True)
    destination = store / snapshot_id
    if destination.exists():
        verify(destination)
        return destination
    staging = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=store))
    try:
        for name, (data, executable) in payloads.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(0o555 if executable else 0o444)
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(canonical_json(manifest))
        manifest_path.chmod(0o444)
        try:
            staging.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            verify(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    verify(destination)
    return destination


def verify(snapshot: Path) -> dict[str, Any]:
    if snapshot.is_symlink():
        raise SnapshotError("Snapshot root must not be a symlink")
    manifest_path = snapshot / "manifest.json"
    if manifest_path.is_symlink():
        raise SnapshotError("Snapshot manifest must not be a symlink")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise SnapshotError("Unsupported snapshot manifest")
    snapshot_id = manifest.get("snapshot_id")
    contract = {key: value for key, value in manifest.items() if key != "snapshot_id"}
    if digest(canonical_json(contract)) != snapshot_id:
        raise SnapshotError("Manifest content does not match its snapshot ID")
    if snapshot.name != snapshot_id:
        raise SnapshotError("Snapshot directory does not match its content address")
    records = manifest.get("files")
    if not isinstance(records, dict):
        raise SnapshotError("Invalid snapshot file records")
    actual = set()
    for path in snapshot.rglob("*"):
        if path.is_symlink():
            raise SnapshotError(f"Snapshot contains a symlink: {path}")
        if not path.is_file() and not path.is_dir():
            raise SnapshotError(f"Snapshot contains a non-regular entry: {path}")
        if path.is_file() and path != manifest_path:
            actual.add(path.relative_to(snapshot).as_posix())
    if actual != set(records):
        raise SnapshotError("Snapshot file set differs from its manifest")
    for name, record in records.items():
        path = snapshot / name
        data = path.read_bytes()
        if (digest(data) != record["sha256"] or len(data) != record["bytes"]
                or bool(path.stat().st_mode & stat.S_IXUSR) != record["executable"]):
            raise SnapshotError(f"Snapshot file changed: {name}")
    return manifest


def clone_recipe(repo: Path, source: Path, name: str) -> Path:
    repo = repo.resolve()
    source = _recipe(repo, source)
    if not RECIPE_NAME.fullmatch(name):
        raise SnapshotError("Recipe names use lowercase letters, digits, and underscores")
    destination = source.parent / name
    if destination.exists():
        raise SnapshotError(f"Recipe already exists: {destination}")
    paths = _source_files(source)
    payloads = {path.relative_to(source).as_posix(): (path.read_bytes(), bool(path.stat().st_mode & 0o111)) for path in paths}
    metadata = read_json(source / "recipe.json")
    if not isinstance(metadata, dict):
        raise SnapshotError("recipe.json must be an object")
    metadata = {**metadata, "id": name, "parent": {
        "recipe": source.relative_to(repo).as_posix(),
        "source_sha256": digest(canonical_json({key: {"sha256": digest(value), "executable": executable}
                                               for key, (value, executable) in payloads.items()})),
    }}
    staging = Path(tempfile.mkdtemp(prefix=".clone-", dir=source.parent))
    try:
        for relative, (data, executable) in payloads.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(0o755 if executable else 0o644)
        (staging / "recipe.json").write_bytes(canonical_json(metadata))
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination
