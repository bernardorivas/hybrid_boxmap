from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from hybrid_dynamics.src import io_utils
from hybrid_dynamics.src.io_utils import atomic_write_json


def _temporary_artifacts(target: Path) -> list[Path]:
    return list(target.parent.glob(f".{target.name}.*.tmp"))


@pytest.mark.parametrize("binary", [False, True])
def test_atomic_write_json_creates_parent_and_writes_strict_utf8_json(
    tmp_path: Path,
    binary: bool,
) -> None:
    target = tmp_path / "nested" / "artifact.json"

    atomic_write_json(target, {"z": 1, "a": "β"}, binary=binary)

    assert target.read_bytes() == (
        b'{\n  "a": "\\u03b2",\n  "z": 1\n}\n'
    )
    assert _temporary_artifacts(target) == []


def test_atomic_write_json_replaces_an_existing_file_by_default(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.json"
    target.write_text("old\n", encoding="utf-8")

    atomic_write_json(target, {"new": True})

    assert target.read_text(encoding="utf-8") == '{\n  "new": true\n}\n'
    assert _temporary_artifacts(target) == []


def test_atomic_write_json_can_refuse_an_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    target.write_text("old\n", encoding="utf-8")

    with pytest.raises(FileExistsError) as exc_info:
        atomic_write_json(target, {"new": True}, refuse_existing=True)

    assert str(exc_info.value) == f"refusing to overwrite {target}"
    assert target.read_text(encoding="utf-8") == "old\n"
    assert _temporary_artifacts(target) == []


def test_atomic_write_json_rejects_nan_and_cleans_up_temporary_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / "artifact.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        atomic_write_json(target, {"invalid": float("nan")})

    assert target.parent.is_dir()
    assert not target.exists()
    assert _temporary_artifacts(target) == []


def test_atomic_write_json_cleans_up_when_replacement_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.json"
    target.write_text("old\n", encoding="utf-8")
    replacement_source: Path | None = None

    def fail_replacement(source: Path, destination: Path) -> None:
        nonlocal replacement_source
        replacement_source = Path(source)
        assert destination == target
        assert replacement_source.read_text(encoding="utf-8") == (
            '{\n  "new": true\n}\n'
        )
        raise OSError("replacement failed")

    monkeypatch.setattr(io_utils.os, "replace", fail_replacement)

    with pytest.raises(OSError, match="replacement failed"):
        atomic_write_json(target, {"new": True})

    assert target.read_text(encoding="utf-8") == "old\n"
    assert replacement_source is not None
    assert not replacement_source.exists()
    assert _temporary_artifacts(target) == []


def test_atomic_write_json_default_mode_does_not_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.json"

    def unexpected_fsync(_descriptor: int) -> None:
        pytest.fail("default atomic JSON writes must not fsync")

    monkeypatch.setattr(io_utils.os, "fsync", unexpected_fsync)

    atomic_write_json(target, {"durable": False})

    assert target.exists()


def test_atomic_write_json_durable_mode_syncs_file_and_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.json"
    synced_file_types: list[int] = []

    def record_fsync(descriptor: int) -> None:
        synced_file_types.append(stat.S_IFMT(os.fstat(descriptor).st_mode))

    monkeypatch.setattr(io_utils.os, "fsync", record_fsync)

    atomic_write_json(target, {"durable": True}, durable=True)

    assert synced_file_types == [stat.S_IFREG, stat.S_IFDIR]
    assert target.read_text(encoding="utf-8") == (
        '{\n  "durable": true\n}\n'
    )
    assert _temporary_artifacts(target) == []


def test_atomic_write_json_cleans_up_when_file_sync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.json"

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("sync failed")

    monkeypatch.setattr(io_utils.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="sync failed"):
        atomic_write_json(target, {"durable": True}, durable=True)

    assert not target.exists()
    assert _temporary_artifacts(target) == []
