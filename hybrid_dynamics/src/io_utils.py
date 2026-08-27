"""Small, dependency-free helpers for writing durable artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def atomic_write_json(
    path: Path,
    value: object,
    *,
    refuse_existing: bool = False,
    durable: bool = False,
    binary: bool = False,
) -> None:
    """Write strict, deterministic JSON and atomically replace ``path``.

    ``refuse_existing`` retains the callers' preflight overwrite check; it is
    intentionally not an exclusive-create guarantee.  ``durable`` additionally
    syncs the temporary file before replacement and the parent directory after
    replacement.  ``binary`` preserves callers that wrote the UTF-8 payload via
    :meth:`Path.write_bytes` rather than text mode.
    """

    if refuse_existing and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        rendered = (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
        if binary:
            temporary.write_bytes(rendered.encode("utf-8"))
        else:
            temporary.write_text(rendered, encoding="utf-8")

        if durable:
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())

        os.replace(temporary, path)

        if durable:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
