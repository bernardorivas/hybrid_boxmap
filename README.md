# hybrid_dynamics

Code and stored results for the paper "Hybrid Attractor Lattices" by William
Kalies, Bernardo Rivas, and Tony Wehbe.

A hybrid system $\mathcal H=(X,\varphi,G,r)$ is studied through its suspension
semiflow $\Phi$ on $\Sigma X$ and the time-$\tau$ map $f_\tau$. For each
example of the paper, the code builds the suspension grid $\Xi_n$ on
$\Sigma(R)$, where $R$ is a rectangle (for the spiking neuron, the L-shaped
set $X$). It then samples a multivalued map
$\mathcal F:\Xi_n\rightrightarrows\Xi_n$ and computes:
- the Morse graph of $\mathcal F$;
- the Conley index of each Morse set over $\mathbb F_5$;
- the lattice of down-sets of the Morse sets with nontrivial Conley index.

The image of a grid element is sampled at the four vertices of each of its
rectangular pieces, and the atoms that contain the sampled images are padded
by one atom. An edge of a piece is bisected, up to 14 times, where the images
of its endpoints lie in atoms whose closures are disjoint. Images that leave
$\Sigma(R)$ are discarded. As stated in the paper, the resulting maps are not
rigorous outer approximations.

## Installation

Python 3.12 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The Conley indices use a fork of [CMGDB](https://github.com/bernardorivas/CMGDB)
at a fixed commit. Building it needs a C++ compiler, Boost 1.56 or newer, GMP,
and SDSL v3; see the fork's installation notes. CMGDB also imports the
`graphviz` Python package.

```bash
python -m pip install --force-reinstall --no-deps --no-cache-dir \
  "git+https://github.com/bernardorivas/CMGDB.git@4150b75304a9b70e5656d46838353a18041a8beb"
python -m pip install graphviz
python -c "import CMGDB; print(hasattr(CMGDB, 'ComputeRelativeHomologyShiftClass'))"   # True
```

The figures lay out Morse graphs and lattices with the Graphviz `dot` program
when it is on the `PATH` (for example `brew install graphviz`), and fall back
to a simpler layout otherwise.

## Reproducing the figures of the paper

`figures/paper_grid/` holds the stored results of the four runs used in the
paper. The commands below write a JSON record and its figures to that folder
and replace the stored results. Add `--output-dir DIR` to write them elsewhere.

| Example | Command | Time |
|---|---|---|
| Bouncing ball, $\tau=0.5$, $2^{10}$ base cells per axis | `python demo/run_paper_grid_examples.py bouncing-ball --tau bouncing-ball=0.5 --level bouncing-ball=6 --level-offset bouncing-ball=4 --gap-refinement-depth 14 --workers 12` | 2 min |
| Rimless wheel, $\tau=0.5$, $2^{11}$ | `python demo/run_paper_grid_examples.py rimless-wheel --tau rimless-wheel=0.5 --level rimless-wheel=7 --level-offset rimless-wheel=4 --gap-refinement-depth 14 --workers 12` | 9 min |
| Spiking neuron, $\tau=5$, $2^{10}$ | `python demo/run_paper_grid_examples.py spiking-neuron --tau spiking-neuron=5 --level spiking-neuron=7 --level-offset spiking-neuron=3 --gap-refinement-depth 14 --workers 12` | 2 min |
| Impacting oscillator, $\beta=0.76$, $\tau=0.5$, $2^{11}$ | `python demo/run_paper_grid_examples.py impact-vdp-duffing-beta076 --tau impact-vdp-duffing-beta076=0.5 --level impact-vdp-duffing-beta076=7 --level-offset impact-vdp-duffing-beta076=4 --gap-refinement-depth 14 --workers 12 --index-workers 3` | 28 min |

The times were measured with 12 workers on a 14-core machine. Memory use
depends on the example:
- the ball and the neuron need a few GB;
- the wheel needs about 15 GB;
- the largest process of the oscillator run reached 23 GB.

The stored oscillator run was made before the runner could compute the Conley
index of $M(21)$ itself. That index was computed afterwards, in 12 minutes
with a peak of 18 GB:

```bash
python demo/fill_missing_labels.py --workers 4 \
  figures/paper_grid/paper-grid-impact-vdp-duffing-beta076-tau050-level7-base2048-corners-gap-refined.json
```

The current runner computes this index within the run (`--index-map auto`,
the default). A rerun therefore takes longer and needs more memory than the
times above.

To redraw the figures from the stored records without recomputing the
dynamics and without changing the records:

```bash
python demo/replot_paper_grid.py --no-update-json --output-dir DIR figures/paper_grid/paper-grid-*.json
```

Each record stores:
- the parameters, the grid, and the image rule;
- the Morse sets and the Morse graph;
- the Conley index of every Morse set;
- the command line and the code commit.

[figures/paper_grid/README.md](figures/paper_grid/README.md) lists which
figure files the paper includes.

## From the paper to the code

| Paper | Code |
|---|---|
| Suspension grid $\Xi_n$, base grids $\mathcal X_n$, the map $d_n$, and $\rho_n=d_n^{-1}$ | `hybrid_dynamics/src/suspension_grid.py` |
| Semiflow $\Phi$ and $f_\tau$ | `hybrid_dynamics/src/suspension_grid_relation.py`, and `hybrid_dynamics/src/batched_suspension_flow.py` for many initial points at once |
| Sampled multivalued map $\mathcal F$ and Morse graph $\mathrm{MG}(\mathcal F)$ | `hybrid_dynamics/src/suspension_grid_relation.py` |
| Conley index of a Morse set: index pair, relative homology over $\mathbb F_5$, index map, shift equivalence class | `hybrid_dynamics/src/suspension_grid_conley.py` and `hybrid_dynamics/src/atlas_conley.py`, with CMGDB's `ComputeRelativeHomologyShiftClass` |
| Lattice of down-sets and its join-irreducible elements | `hybrid_dynamics/src/attractor_lattice.py` |
| The four examples | `hybrid_dynamics/examples/paper_grid_examples.py` |
| Figures | `hybrid_dynamics/examples/paper_grid_figures.py` |

[PAPER_GRID.md](PAPER_GRID.md) documents the implementation statement by
statement, with the tests that check each part. [proofs.md](proofs.md)
collects the arguments behind the descriptions of the four examples and
compares them with the stored runs.

## Tests

```bash
python -m pytest -q
```

The test suite needs the CMGDB fork. It has 421 tests and takes about 2.5
minutes. One test reads a large local file that is not in the repository; it
is skipped when the file is absent.

## Repository layout

- `hybrid_dynamics/`: the package, with its tests in `hybrid_dynamics/tests/`.
- `demo/`: the command-line runners above, and one audit script used by a test.
- `figures/paper_grid/`: the stored records and figures of the paper's four runs.
- `data/`: stored relations and baselines, some of them read by the tests.
- `test/`: additional tests.
- `docs/library.md`: the general library (hybrid trajectories, box maps, the
  earlier tagged-chart pipeline).
- `archive/`: earlier examples, demos, figures, experiments, and notes, kept
  for reference. They are not needed for the paper.

## Citation

```bibtex
@misc{Kalies:Rivas:Wehbe,
  author = {Kalies, William and Rivas, Bernardo and Wehbe, Tony},
  title  = {Hybrid Attractor Lattices},
  year   = {2026},
  note   = {Preprint}
}
```

## License

MIT; see [LICENSE](LICENSE).
