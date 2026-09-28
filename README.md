# hybrid_dynamics

This package extends [CMGDB](https://github.com/marciogameiro/CMGDB) to a class
of hybrid dynamical systems. It constructs combinatorial outer approximations
of the dynamics from sampled trajectories, and computes their Morse graphs,
Conley indices, and lattices of attractors.

Hybrid systems are handled through their suspension, a continuous semiflow
obtained by attaching a unit time interval at each point of the guard.

## Installation

Python 3.12 or newer is required.

```bash
git clone https://github.com/bernardorivas/hybrid_boxmap.git
cd hybrid_boxmap
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The Conley indices use a fork of CMGDB. Building it requires a C++ compiler,
Boost, GMP, and SDSL.

```bash
python -m pip install --no-deps \
  "git+https://github.com/bernardorivas/CMGDB.git@4150b75304a9b70e5656d46838353a18041a8beb"
python -m pip install graphviz
```

## Usage

The following example computes the Morse sets of the bouncing ball and their
Conley indices. It takes about a minute.

```python
from hybrid_dynamics.examples.paper_grid_examples import paper_grid_problem
from hybrid_dynamics.src.suspension_grid import build_suspension_grid
from hybrid_dynamics.src.suspension_grid_relation import (
    compute_suspension_grid_relation,
    compute_suspension_morse_graph,
)
from hybrid_dynamics.src.suspension_grid_conley import (
    compute_suspension_grid_conley_indices,
)

problem = paper_grid_problem("bouncing-ball", tau=0.5, level_offset=3)
grid = build_suspension_grid(problem.window, problem.guard, level=5)
relation = compute_suspension_grid_relation(grid, problem, gap_refinement_depth=14)
morse_graph = compute_suspension_morse_graph(relation)
indices = compute_suspension_grid_conley_indices(
    relation, morse_graph.morse_sets, index_map="auto"
)
for index in indices:
    print(index.morse_node, index.shift_class)
```

Other examples are the rimless wheel, a spiking neuron, and an impacting
oscillator. The command-line runner computes any of them and saves the results
and figures:

```bash
python demo/run_paper_grid_examples.py --help
```

## Results in the paper

The results of the paper "Hybrid Attractor Lattices" and the commands that
produce them are in [`figures/paper_grid/`](figures/paper_grid/README.md).

## Tests

```bash
python -m pytest -q
```

## Citation

```bibtex
@software{Rivas:hybrid_dynamics,
  author = {Rivas, Bernardo},
  title  = {hybrid\_dynamics},
  url    = {https://github.com/bernardorivas/hybrid_boxmap},
  year   = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
