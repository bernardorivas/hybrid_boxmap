# Results in the paper

Records and figures of the four computations in the paper "Hybrid Attractor
Lattices". Each record (`.json`) stores the parameters, the Morse graph, the
Conley indices, and the command that produced it. Its figures share its name.

## Commands

Run from the repository root. Times are for 12 workers.

| Example | Command | Time |
|---|---|---|
| Bouncing ball | `python demo/run_paper_grid_examples.py bouncing-ball --tau bouncing-ball=0.5 --level bouncing-ball=6 --level-offset bouncing-ball=4 --gap-refinement-depth 14 --workers 12` | 2 min |
| Rimless wheel | `python demo/run_paper_grid_examples.py rimless-wheel --tau rimless-wheel=0.5 --level rimless-wheel=7 --level-offset rimless-wheel=4 --gap-refinement-depth 14 --workers 12` | 9 min |
| Spiking neuron | `python demo/run_paper_grid_examples.py spiking-neuron --tau spiking-neuron=5 --level spiking-neuron=7 --level-offset spiking-neuron=3 --gap-refinement-depth 14 --workers 12` | 2 min |
| Impacting oscillator | `python demo/run_paper_grid_examples.py impact-vdp-duffing-beta076 --tau impact-vdp-duffing-beta076=0.5 --level impact-vdp-duffing-beta076=7 --level-offset impact-vdp-duffing-beta076=4 --gap-refinement-depth 14 --workers 12 --index-workers 3` | about 30 min |

The oscillator needs more than 20 GB of memory. Add `--output-dir DIR` to keep
the stored results.

To redraw the figures from the stored records:

```bash
python demo/replot_paper_grid.py --no-update-json --output-dir DIR figures/paper_grid/paper-grid-*.json
```
