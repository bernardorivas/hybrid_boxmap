#!/usr/bin/env python3
"""Draw the spiking-neuron Morse-set figure with the shared Atlas plotter.

The accepted terminal-carrier stage is loaded through the strict Conley input
reader (CSR relation, authenticated provenance, and stage summary all
cross-checked), projected onto ``AtlasMorsePlotData``, and annotated with the
finite-relation shift class through the same gated loader the bouncing-ball
and rimless-wheel diagnostics use.  The figure therefore has the same layout,
palette, and labeling conventions as those two.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from hybrid_dynamics import (
    PlotHybridMorseSets,
    load_atlas_finite_relation_index_annotations,
    save_atlas_morse_plot_data,
    save_hybrid_morse_figure,
)
from hybrid_dynamics.examples.spiking_neuron_conley import (
    load_spiking_neuron_conley_input,
    spiking_neuron_atlas_morse_plot_data,
    spiking_neuron_finite_relation_index_audit,
    validate_spiking_neuron_conley_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_STEM = "hybrid-morse-spiking-neuron-atlas-tau2000-depth16"


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
        "--paper-dir",
        type=Path,
        default=PROJECT_ROOT / "paper" / "figures",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/paper_figure_manifests"),
    )
    parser.add_argument(
        "--include-handles",
        action="store_true",
        help="add the intrinsic guard-coordinate/phase chart panel",
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


def main() -> int:
    args = _arguments()
    primary = args.scientific_dir / "adaptive_terminal_bridge_v1/t20"
    sensitivity = args.scientific_dir / "adaptive_terminal_bridge_samples5_v1/t20"
    primary_conley_path = primary / "conley_v2/finite_relation_conley.json"
    sensitivity_conley_path = sensitivity / "conley_v2/finite_relation_conley.json"
    validate_spiking_neuron_conley_summary(primary_conley_path)
    validate_spiking_neuron_conley_summary(sensitivity_conley_path)
    comparison = _validate_comparison(
        args.scientific_dir / "adaptive_terminal_bridge_samples5_v1/comparison.json"
    )
    conley = json.loads(primary_conley_path.read_text(encoding="utf-8"))

    conley_input = load_spiking_neuron_conley_input(primary)
    if conley_input.relation_reference["fingerprint"] != (
        conley["input_artifacts"]["relation_csr"]["fingerprint"]
    ):
        raise ValueError("stage relation and Conley summary are bound to different CSRs")
    data = spiking_neuron_atlas_morse_plot_data(conley_input)
    cache_path = primary / "morse_plot_tau2000_depth16.json"
    save_atlas_morse_plot_data(data, cache_path)

    audit_path = primary / "conley_v2/finite_relation_index_audit.json"
    audit_path.write_text(
        json.dumps(
            spiking_neuron_finite_relation_index_audit(primary_conley_path),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    annotations = load_atlas_finite_relation_index_annotations(audit_path, data)

    plot = PlotHybridMorseSets(
        data,
        finite_relation_annotations=annotations,
        axis_labels=(r"$v$", r"$u$"),
        handle_axis_labels=(r"$u_G$", r"$s$"),
        show_handles=args.include_handles,
        show_morse_graph=True,
        show_legend=False,
        show_panel_titles=False,
        show_component_sizes=False,
        show_status_note=False,
        base_view="support",
        fig_h=3.4,
    )
    try:
        outputs = save_hybrid_morse_figure(
            plot,
            args.output_dir / FIGURE_STEM,
            formats=("pdf", "png"),
            dpi=400,
        )
    finally:
        plt.close(plot.figure)
    pdf_path = next(path for path in outputs if path.suffix == ".pdf")
    png_path = next(path for path in outputs if path.suffix == ".png")
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    paper_copy = args.paper_dir / pdf_path.name
    paper_copy.write_bytes(pdf_path.read_bytes())

    manifest = {
        "schema": "spiking-neuron-finite-relation-figure-v2",
        "model": data.metadata["model"],
        "depth": data.metadata["depth"],
        "t_star": data.metadata["t_star"],
        "plot_cache": str(cache_path),
        "finite_relation_audit": str(audit_path),
        "primary_conley_fingerprint": conley["fingerprint"],
        "sensitivity_conley_fingerprint": json.loads(
            sensitivity_conley_path.read_text(encoding="utf-8")
        )["fingerprint"],
        "sampling_comparison_fingerprint": comparison["fingerprint"],
        "primary_relation_csr_fingerprint": conley_input.relation_reference[
            "fingerprint"
        ],
        "morse_nodes": list(data.vertex_ids),
        "morse_edges": [list(edge) for edge in data.edges],
        "morse_set_cell_counts": {
            str(node.index): len(node.boxes) for node in data.nodes
        },
        "finite_relation_conley_index": {
            "computed": True,
            "coefficient_field": 5,
            "result_scope": annotations.result_scope,
            "shift_classes": {
                str(node): list(entries)
                for node, entries in annotations.shift_classes.items()
            },
        },
        "continuous_system_conley_index_certified": False,
        "pdf": str(pdf_path),
        "pdf_sha256": _sha256_file(pdf_path),
        "png": str(png_path),
        "png_sha256": _sha256_file(png_path),
        "paper_copy": str(paper_copy),
        "paper_copy_sha256": _sha256_file(paper_copy),
    }
    manifest["fingerprint"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    (args.manifest_dir / "spiking_neuron_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
