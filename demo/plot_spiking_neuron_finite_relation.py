#!/usr/bin/env python3
"""Create the strict-report-backed vector diagnostic for the neuron example."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from hybrid_dynamics.examples.spiking_neuron_atlas import (
    compute_neuron_reference_cycle,
    validate_spiking_neuron_provenance,
)
from hybrid_dynamics.examples.spiking_neuron_conley import (
    validate_spiking_neuron_conley_summary,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scientific-dir",
        type=Path,
        default=Path("data/spiking_neuron_atlas/scientific_clock_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pdf"),
    )
    parser.add_argument(
        "--paper-figure",
        type=Path,
        default=Path(
            "../paper/figures/atlas-diagnostics/"
            "spiking_neuron_finite_relation_diagnostic.pdf"
        ),
    )
    return parser.parse_args()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_comparison(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.get("fingerprint")
    unsigned = dict(payload)
    unsigned.pop("fingerprint", None)
    if hashlib.sha256(_canonical(unsigned)).hexdigest() != claimed:
        raise ValueError("sampling-comparison fingerprint mismatch")
    if not payload.get("sampling_persistence_passed"):
        raise ValueError("sampling persistence did not pass")
    if payload.get("continuous_system_conley_index_certified") is not False:
        raise ValueError("comparison overstates continuous certification")
    return payload


def _morse_rectangles(
    provenance: Path,
    *,
    expected_relation_csr_fingerprint: str,
    expected_source_count: int,
    expected_provenance_reference: dict[str, object],
) -> tuple[dict[int, list[tuple[float, ...]]], dict[str, object]]:
    authenticated = validate_spiking_neuron_provenance(
        provenance,
        expected_relation_csr_fingerprint=expected_relation_csr_fingerprint,
    )
    if authenticated != expected_provenance_reference:
        raise ValueError(
            "plotted provenance does not match the strict Conley input reference"
        )
    if authenticated["source_records"] != expected_source_count:
        raise ValueError("plotted provenance source count is inconsistent")

    result: dict[int, list[tuple[float, ...]]] = {0: [], 1: []}
    seen_sources = 0
    with gzip.open(provenance, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if not header["stage_gates_passed"]:
            raise ValueError("figure source relation did not pass its gates")
        if header.get("active_cells") != expected_source_count:
            raise ValueError("figure source header has the wrong source count")
        if (
            header.get("relation_csr", {}).get("fingerprint")
            != expected_relation_csr_fingerprint
        ):
            raise ValueError("figure source header is bound to a different relation")
        for line in stream:
            record = json.loads(line)
            if record["record"] == "trailer":
                break
            if record["record"] != "source":
                raise ValueError("unknown figure provenance record")
            seen_sources += 1
            if record["morse_node"] == 0:
                result[int(record["chart_id"])].append(
                    tuple(float(value) for value in record["bounds"])
                )
    if seen_sources != expected_source_count:
        raise ValueError("figure source stream is incomplete")
    return result, authenticated


def _add_rectangles(
    axis: plt.Axes,
    rectangles: list[tuple[float, ...]],
    *,
    facecolor: str,
    edgecolor: str,
) -> None:
    for x0, y0, x1, y1 in rectangles:
        axis.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=0.22,
                alpha=0.72,
                rasterized=False,
            )
        )


def _domain_inset(axis: plt.Axes) -> None:
    inset = axis.inset_axes([0.72, 0.58, 0.25, 0.36])
    domain = np.asarray(
        [
            (-80.0, -300.0),
            (35.0, -300.0),
            (35.0, 160.0),
            (-40.0, 160.0),
            (-40.0, 600.0),
            (-80.0, 600.0),
        ]
    )
    inset.add_patch(
        Polygon(
            domain,
            closed=True,
            facecolor="#eceff1",
            edgecolor="#263238",
            linewidth=1.0,
        )
    )
    inset.add_patch(
        Rectangle(
            (-67.0, -85.0),
            105.0,
            190.0,
            fill=False,
            edgecolor="#c62828",
            linewidth=1.1,
        )
    )
    inset.set_xlim(-86, 41)
    inset.set_ylim(-340, 640)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("#78909c")
        spine.set_linewidth(0.7)


def _handle_inset(axis: plt.Axes) -> None:
    inset = axis.inset_axes([0.72, 0.60, 0.25, 0.33])
    inset.add_patch(
        Rectangle(
            (-300.0, 0.0),
            460.0,
            1.0,
            facecolor="#fff3e0",
            edgecolor="#263238",
            linewidth=1.0,
        )
    )
    inset.add_patch(
        Rectangle(
            (-75.0, 0.0),
            70.0,
            1.0,
            fill=False,
            edgecolor="#c62828",
            linewidth=1.1,
        )
    )
    inset.set_xlim(-330, 190)
    inset.set_ylim(-0.08, 1.08)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("#78909c")
        spine.set_linewidth(0.7)


def main() -> int:
    args = _arguments()
    primary = args.scientific_dir / "adaptive_terminal_bridge_v1/t20"
    sensitivity = (
        args.scientific_dir / "adaptive_terminal_bridge_samples5_v1/t20"
    )
    primary_conley_path = primary / "conley_v2/finite_relation_conley.json"
    sensitivity_conley_path = sensitivity / "conley_v2/finite_relation_conley.json"
    validate_spiking_neuron_conley_summary(primary_conley_path)
    validate_spiking_neuron_conley_summary(sensitivity_conley_path)
    conley = json.loads(primary_conley_path.read_text(encoding="utf-8"))
    comparison = _validate_comparison(
        args.scientific_dir
        / "adaptive_terminal_bridge_samples5_v1/comparison.json"
    )
    relation_reference = conley["input_artifacts"]["relation_csr"]
    expected_provenance = conley["input_artifacts"]["source_provenance"]
    rectangles, plotted_provenance = _morse_rectangles(
        primary / "provenance.jsonl.gz",
        expected_relation_csr_fingerprint=relation_reference["fingerprint"],
        expected_source_count=int(relation_reference["vertices"]),
        expected_provenance_reference=expected_provenance,
    )
    cycle = compute_neuron_reference_cycle()
    times = np.linspace(0.0, cycle.flight_time, 1400)
    orbit = np.asarray(cycle.solution(times), dtype=np.float64)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, (base_axis, handle_axis) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.9),
        gridspec_kw={"wspace": 0.22},
    )
    figure.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.31)

    _add_rectangles(
        base_axis,
        rectangles[0],
        facecolor="#90caf9",
        edgecolor="#1976d2",
    )
    base_axis.plot(orbit[0], orbit[1], color="#111111", linewidth=1.7, zorder=5)
    base_axis.plot(
        [35.0, -50.0],
        [cycle.pre_reset_u, cycle.post_reset_u],
        color="#111111",
        linewidth=1.25,
        linestyle=(0, (4, 2)),
        zorder=5,
    )
    base_axis.axvline(-50.0, color="#6a1b9a", linewidth=0.9, alpha=0.8)
    base_axis.axvline(35.0, color="#c62828", linewidth=0.9, alpha=0.8)
    base_axis.text(-49.0, 98.0, r"reset seam $v=-50$", color="#6a1b9a", fontsize=7.5)
    base_axis.text(34.0, 98.0, r"guard", color="#c62828", fontsize=7.5, ha="right")
    base_axis.set_xlim(-67.0, 38.0)
    base_axis.set_ylim(-85.0, 105.0)
    base_axis.set_xlabel(r"voltage $v$")
    base_axis.set_ylabel(r"recovery $u$")
    base_axis.grid(color="#cfd8dc", linewidth=0.45, alpha=0.6)
    _domain_inset(base_axis)

    _add_rectangles(
        handle_axis,
        rectangles[1],
        facecolor="#ffcc80",
        edgecolor="#ef6c00",
    )
    handle_axis.axvline(
        cycle.pre_reset_u,
        color="#111111",
        linewidth=1.7,
        zorder=5,
    )
    handle_axis.scatter(
        [cycle.pre_reset_u, cycle.pre_reset_u],
        [0.0, 1.0],
        s=13,
        color="#111111",
        zorder=6,
    )
    handle_axis.set_xlim(-75.0, -5.0)
    handle_axis.set_ylim(-0.02, 1.02)
    handle_axis.set_xlabel(r"guard coordinate $u_g$")
    handle_axis.set_ylabel(r"handle phase $s$")
    handle_axis.grid(color="#cfd8dc", linewidth=0.45, alpha=0.6)
    _handle_inset(handle_axis)

    shift = ", ".join(conley["finite_relation_shift_class"][:2])
    figure.text(
        0.5,
        0.185,
        (
            r"Finite relation over $\mathbb{F}_5$: "
            r"$H_0\cong H_1\cong\mathbb{F}_5$, $F_{*0}=F_{*1}=1$; "
            rf"shift class $({shift})$"
        ),
        ha="center",
        fontsize=9.1,
        color="#0d47a1",
    )
    figure.text(
        0.5,
        0.092,
        (
            "Sampled terminal-carrier model only: no rigorous whole-cell outer "
            "enclosure and no certified continuous-system Conley index."
        ),
        ha="center",
        fontsize=8.1,
        color="#b71c1c",
    )
    figure.text(
        0.985,
        0.018,
        f"strict comparison {comparison['fingerprint'][:12]}...",
        ha="right",
        fontsize=5.8,
        color="#607d8b",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "spiking_neuron_finite_relation_diagnostic.pdf"
    png_path = args.output_dir / "spiking_neuron_finite_relation_diagnostic.png"
    figure.savefig(pdf_path, format="pdf", metadata={"CreationDate": None})
    figure.savefig(png_path, format="png", dpi=260)
    plt.close(figure)
    args.paper_figure.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf_path, args.paper_figure)
    manifest = {
        "schema": "spiking-neuron-finite-relation-figure-v1",
        "primary_conley_fingerprint": conley["fingerprint"],
        "sensitivity_conley_fingerprint": json.loads(
            sensitivity_conley_path.read_text(encoding="utf-8")
        )["fingerprint"],
        "sampling_comparison_fingerprint": comparison["fingerprint"],
        "primary_relation_csr_fingerprint": conley["input_artifacts"][
            "relation_csr"
        ]["fingerprint"],
        "primary_source_provenance": plotted_provenance,
        "finite_relation_only": True,
        "continuous_system_conley_index_certified": False,
        "pdf": pdf_path.name,
        "pdf_sha256": _sha256_file(pdf_path),
        "png": png_path.name,
        "png_sha256": _sha256_file(png_path),
        "paper_copy": str(args.paper_figure),
        "paper_copy_sha256": _sha256_file(args.paper_figure),
    }
    manifest["fingerprint"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    (args.output_dir / "spiking_neuron_finite_relation_diagnostic.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
