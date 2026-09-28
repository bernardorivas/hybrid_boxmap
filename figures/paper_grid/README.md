# Paper-grid runs

The four runs used in the paper, one JSON record each, with their figures under
the same name. The commands that produce them are in the
[README](../../README.md#reproducing-the-figures-of-the-paper).

| Example | Record |
|---|---|
| Bouncing ball | `paper-grid-bouncing-ball-tau050-level6-base1024-corners-gap-refined.json` |
| Rimless wheel | `paper-grid-rimless-wheel-tau050-level7-base2048-corners-gap-refined.json` |
| Spiking neuron | `paper-grid-spiking-neuron-tau500-level7-base1024-corners-gap-refined.json` |
| Impacting oscillator, $\beta=0.76$ | `paper-grid-impact-vdp-duffing-beta076-tau050-level7-base2048-corners-gap-refined.json` |

For a record `<stem>.json`:

- `<stem>.pdf` shows every Morse set, with the Morse sets of trivial Conley
  index in gray, and `<stem>-nontrivial.pdf` shows only those with nontrivial
  Conley index.
- `-base`, `-zoom-A`, `-zoom-B`, ..., and `-graph` are the panels of these two
  figures as separate files, for both variants. `-attractor-lattice` is the
  lattice of down-sets of the Morse sets with nontrivial Conley index.
- For every example, the paper includes the `-graph` panel (the full
  Conley–Morse graph) and the `-nontrivial-base` panel. It also includes the
  `-nontrivial-zoom-*` panels for the ball, the wheel, and the oscillator, the
  `-nontrivial-graph` panel (the restricted Conley–Morse graph) for the wheel
  and the oscillator, and the `-attractor-lattice` figure for the oscillator.

The record stores:
- the parameters, the grid, and the image rule;
- the Morse sets and the Morse graph;
- the Conley index of every Morse set;
- the command line and the code commit.

`demo/replot_paper_grid.py` redraws the figures from a record and also writes
PNG copies, which are not tracked. The Conley index of $M(21)$ in the
oscillator record was computed afterwards by `demo/fill_missing_labels.py`,
with the index map built by excision. The record lists that command under
`labels_filled`.
