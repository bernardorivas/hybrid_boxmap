# Paper-grid runs

Outputs of `demo/run_paper_grid_examples.py`, described in `PAPER_GRID.md`.
Each run has a JSON summary and its figures under the same stem.

- `./`: the recommended run of each example for the paper figures, with
  `<stem>.pdf/.png` (every Morse node) and `<stem>-nontrivial.pdf/.png`
  (trivial-index nodes hidden). The ball, the wheel, and the neuron at
  `tau = 5` were recomputed at `a47c88d` (batched integrator, relative
  homology of every Morse set, no piece limit); the oscillator at
  `beta = 0.8` is the run at `8e99989`, with figures redrawn at `6441c93`.
  The neuron run at `tau = 1` that the sweep first picked is also kept, from
  `8e99989`.
- `./`, variant `impact-vdp-duffing-beta076` (the oscillator at
  `beta = 0.76`, same window): the run at `tau = 1`, level 6, 1024 base
  cells and the run at `tau = 0.5`, level 7, 2048 base cells, both
  gap-refined, recomputed at `a47c88d` (first run at `a7ac462`), each with
  both figure variants. In both runs the label of U_Z, `(0, x-1, x-1, 0)`
  from the excision pair, was filled by `demo/fill_missing_labels.py`:
  `M(14)` of the `tau = 1` run at `657bf2a`, and `M(21)` of the
  `tau = 0.5` run in commit `02270ea` (with `--workers 4`, peak 18.3 GB).
  Each fill replaced only that index record (recorded under
  `labels_filled`) and redrew the figures and panels of the run.
  `proofs.md` in `code/` compares the `tau = 0.5` run with the known
  invariant sets and indices.
- Panel figures in `./`: each panel of the two figures of a run is also its
  own figure, PDF and PNG, `<variant stem>-base`, `<variant stem>-zoom-A`,
  `-zoom-B`, ... (one per zoom), `<variant stem>-graph`, and
  `<variant stem>-handle` when the handle chart is drawn (no current run),
  with `<variant stem>` = `<stem>` or `<stem>-nontrivial`. Each JSON lists
  them under `figure_variants.<variant>.panel_files` and in `figures`. Every
  run in `./` was redrawn at `98ee08d` with `demo/replot_paper_grid.py`,
  which added the panel figures and left the combined figures as they were,
  and again at `eeba2a0`, which gave the zoom panel figures axis labels and
  the dark Morse graph nodes white text (combined and panel figures); the
  base panels are unchanged. Every figure and panel is now drawn in
  CMGDB's default palette, with the Morse sets whose computed index is
  trivial in gray (`#BBBBBB`); `M(i)` otherwise keeps CMGDB color `i`, so a
  set has one color in both variants and all panels. (The runs were drawn
  in Paul Tol's "muted" palette at `c896334` and returned to the CMGDB
  palette at the next commit.) Each JSON lists the palette and the color of
  every Morse node under `figure_variants.<variant>.colors`.
- Every run in `./` was redrawn at `ddaa498`. The Morse graph of each figure
  and panel now shows the index labels, and the `dim H` line of a node
  without a label, in degrees 0, 1, 2 only; every dropped entry was checked
  to be zero (none of the runs has a nonzero entry in degree 3 or higher,
  so no node keeps its full tuple). The `conley` records keep every
  degree, and each variant records the degrees shown under
  `figure_variants.<variant>.index_degrees`. The chart and zoom panels are
  unchanged.
- Attractor lattice in `./`: `<stem>-attractor-lattice.pdf/.png`, the Hasse
  diagram of the lattice of down-sets of the Morse order of the nontrivial
  variant (Birkhoff), drawn at `ddaa498` for every run in `./`. The
  join-irreducible elements `↓M(i)` are in the colors of their Morse sets
  (dashed when `M(i)` has no label), the other elements white, labeled by
  joins, with bottom `0`. Each JSON records it under `attractor_lattice`
  and lists its files at the end of `figures`. Elements (join-irreducible):
  ball `tau = 0.5` 2 (1), wheel `tau = 0.5` 3 (2), neuron `tau = 1` and
  `tau = 5` 2 (1), oscillator `beta = 0.8`, `tau = 1` 15 (6, of which
  `M(3)`, `M(5)`, `M(8)` have no label), oscillator `beta = 0.76` at
  `tau = 1` and `tau = 0.5` 11 (5).
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
