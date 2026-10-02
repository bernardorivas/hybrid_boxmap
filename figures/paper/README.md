# Results in the paper

Records and figures of the four examples in the paper "Hybrid Attractor
Lattices". Each record (`.json`) stores the parameters, the Morse graph, the
Conley indices, and the command that produced it. Its figures share its name.

## Commands

Run from the code repository root. Use a new output directory for each computation.
The base grid has `2**(level + level_offset)` cells per axis; the phase grid has
`2**(level + 2)` cells, independently of the offset. All runs below use corner
sampling and gap refinement depth 14. Times use the worker counts in the
commands. Coarse and oscillator times are measured wall times, including
figures, with the native backend. The fine wheel time is the sum of the stages
in its historical record, computed with a different backend; the neuron time
is an estimate.

| Example | Base cells per axis | Phase cells | Command | Time |
|---|---:|---:|---|---|
| Bouncing ball, coarse | 256 | 64 | `python demo/run_paper_examples.py bouncing-ball --tau bouncing-ball=0.5 --level bouncing-ball=4 --level-offset bouncing-ball=4 --gap-refinement-depth 14 --workers 12 --output-dir output/ball-coarse` | 6.76 s |
| Rimless wheel, coarse | 128 | 512 | `python demo/run_paper_examples.py rimless-wheel --tau rimless-wheel=0.5 --level rimless-wheel=7 --level-offset rimless-wheel=0 --gap-refinement-depth 14 --workers 12 --output-dir output/wheel-coarse` | 19.40 s |
| Rimless wheel, fine | 2048 | 512 | `python demo/run_paper_examples.py rimless-wheel --tau rimless-wheel=0.5 --level rimless-wheel=7 --level-offset rimless-wheel=4 --gap-refinement-depth 14 --workers 12 --index-workers 4 --output-dir output/wheel-fine` | 535.62 s (recorded stages) |
| Spiking neuron | 1024 | 512 | `python demo/run_paper_examples.py spiking-neuron --tau spiking-neuron=5 --level spiking-neuron=7 --level-offset spiking-neuron=3 --gap-refinement-depth 14 --workers 12 --output-dir output/neuron` | about 30 s |
| Impacting oscillator | 2048 | 512 | `python demo/run_paper_examples.py impact-vdp-duffing-beta076 --tau impact-vdp-duffing-beta076=0.5 --level impact-vdp-duffing-beta076=7 --level-offset impact-vdp-duffing-beta076=4 --gap-refinement-depth 14 --workers 8 --index-workers 1 --conley-backend native --output-dir output/oscillator` | 334.71 s |

The fine wheel and the oscillator need about 12 GB of memory. The fine wheel
record is retained from the original computation; the comparison only redraws
it. The coarse ball has one Morse set with label `(1,1,0)`. The coarse and fine
wheel graphs have 20 and 17 nodes, respectively; both retain the orbit
`(1,1,0)`, saddle `(0,1,0)`, and the saddle-to-orbit edge after restricting the
order to the two nontrivial nodes. The original fine ball record is retained.

To redraw the figures from the stored records:

```bash
python demo/replot_paper.py --no-update-json --output-dir DIR figures/paper/paper-*.json
```

For the selected ball's explicit zoom and the wheel comparison's common axes:

```bash
python demo/paper_comparison.py bouncing-ball \
  figures/paper/paper-bouncing-ball-tau050-level4-base256-corners-gap-refined.json \
  --output-dir output/ball-panels

python demo/paper_comparison.py rimless-wheel \
  figures/paper/paper-rimless-wheel-tau050-level7-corners-gap-refined.json \
  figures/paper/paper-rimless-wheel-tau050-level7-base2048-corners-gap-refined.json \
  --output-dir output/wheel-comparison
```

These commands preserve the records and write PDF/PNG versions of the
`-nontrivial-base`, `-nontrivial-zoom-A`, `-nontrivial-graph`, and `-graph` panels,
combined figures, and provenance manifests with relative paths. Panels have no
headings apart from the zoom letter; `--titles` adds parameter headings to the
combined figures. The helper also accepts a single wheel record. The notebooks
call the same functions after loading a stored record or computing a new one.
