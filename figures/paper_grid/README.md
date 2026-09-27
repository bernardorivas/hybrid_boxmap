# Paper-grid runs

Outputs of `demo/run_paper_grid_examples.py`, described in `PAPER_GRID.md`.
Each run has a JSON summary and its figures under the same stem.

- `./`: the recommended run of each example for the paper figures (ball,
  wheel, neuron at `tau = 5`, oscillator), with `<stem>.pdf/.png` (every
  Morse node) and `<stem>-nontrivial.pdf/.png` (trivial-index nodes hidden),
  redrawn at `ff7ca77` with base chart, zoom panels, handle chart, and Morse
  graph; and the neuron run at `tau = 1` that the sweep first picked, with
  figures in the earlier layout (base chart and Morse graph).
- `sweep-bouncing-ball/`, `sweep-rimless-wheel/`, `sweep-spiking-neuron/`,
  `sweep-impact-vdp-duffing/`: the small-`tau` sweep at `8e99989` (136 runs),
  figures in the earlier layout.
  `sweep-spiking-neuron/index-max-pieces-150000/` repeats one run with a
  larger piece limit under the same file name.
- `large_tau/`: corner runs at the manuscript's `tau` (and `tau` 3 to 6.5 for
  the oscillator) at `0706668`, before the seam fix `788063c`.
- `pre_fix/`: small-`tau` corner runs at `0706668`.
- `superseded-tensor3-350c93e/`: the `3 x 3` tensor runs at `350c93e`.

The paths recorded in a JSON file (`figures`, `figure_variants`,
`image_rule.command_line`) are those at write time. Runs that were copied from
scratch directories or moved into a subfolder still list their old paths;
their figures sit next to the JSON. The summaries in `large_tau/` and
`pre_fix/` (schema `paper-suspension-grid-run-v2`) store no Morse-set atoms
and cannot be redrawn with `demo/replot_paper_grid.py`.
