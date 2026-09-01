# Experimental record

This document condenses the useful results from the exploratory repository.
Run directories and checkpoints are intentionally not part of this Git
repository. Metrics from different training regimes are reported separately;
they should not be pooled as if they were matched trials.

## Representation and architecture

The original image-patch encoding was replaced by exact ternary signal cells.
Two operand columns contain `n` cells each and the output column contains
`n + 1` cells, including a leading carry row. Horizontal and vertical strides
and all borders are configurable. Neutral cells have value zero.

The compact state was reduced to three channels. Channel 0 is initialized with
the operands; all three channels then evolve. This removed the earlier
hardcoded, read-only input memory and made both computation and memory part of
the learned dynamics.

The most useful compact perception bank was identity + Sobel-X + Sobel-Y plus
one shared learnable 3x3 kernel initialized as

```text
0     1/4   0
1/4   -1    1/4
0     1/4   0
```

With no write gate and hidden width 57, this model has 921 trainable
parameters. A gated hidden-width-48 model has 927 parameters and was used for
approximately parameter-matched gate comparisons.

## Reference checkpoints

These paths identify local historical artifacts in the exploratory repository;
they are not included here.

| Model | Historical checkpoint | 4-bit supervised-window result |
|---|---|---:|
| No gate, h57, quality-diversity winner | `QD_adder4_cells_c3h57_nogate_learnk_T70_20260829_185944/checkpoints/best.pt` | MSE 0.001289, exact 100% |
| Linear gate, h48, seed 5 | `CELL_adder4_cells_c3h48_wglinear_learnk_s5_20260826_153825/checkpoints/nca_001000.pt` | MSE 0.001848, exact 100% |
| Sigmoid gate, h48, seed 6 | `CELL_adder4_cells_c3h48_wgsigmoid_learnk_s6_20260826_155058/checkpoints/nca_001000.pt` | MSE 0.001911, exact 100% |

## Width extrapolation on steps 71-140

All ordered operand pairs were evaluated.

| Model | Width | Exact averaged over window | Stable over full window | Bit accuracy | MSE |
|---|---:|---:|---:|---:|---:|
| No gate h57 | 6 | 95.98% | 95.24% | 98.99% | 0.03903 |
| No gate h57 | 8 | 91.90% | 90.56% | 98.25% | 0.06732 |
| Linear h48 | 6 | 96.28% | 94.68% | 99.21% | 0.02592 |
| Linear h48 | 8 | 91.84% | 89.26% | 98.51% | 0.04972 |
| Sigmoid h48 | 6 | 95.30% | 94.82% | 99.05% | 0.03004 |
| Sigmoid h48 | 8 | 89.68% | 89.10% | 98.18% | 0.05901 |

## Long-horizon behavior through step 400

The models were trained for supervision only through step 140. Extending the
rollout revealed distinct dynamics:

- The no-gate model peaked at 96.14% exact accuracy on 6-bit addition near
  step 110 and 92.12% on 8-bit addition near step 107, then decayed. Accuracy
  at step 400 was 6.64% and 2.25%, respectively.
- The linear gate reached strong peaks near the trained window but became
  catastrophically unstable after it: 6-bit and 8-bit exact accuracy were both
  approximately zero by steps 300-400.
- The sigmoid gate preserved its computation. At step 400 it retained 95.12%
  exact 6-bit accuracy and 89.25% exact 8-bit accuracy. Stable accuracy across
  every step from 71 through 400 was 94.78% and 88.85%.

Thus gates did not show a reliable general training-accuracy advantage across
matched seeds, and the early linear-gate winner was consistent with seed luck.
The sigmoid gate nevertheless showed a clear dynamical benefit as a
long-horizon stabilizer. These are different claims and should not be conflated.

## Optimization observations

- Exact exhaustive batches reduce gradient noise but did not by themselves
  produce rapid grokking in the very small network.
- Increasing computation time too abruptly made curriculum training harder.
- CMA-ES did not improve the approximately 900-parameter model under the tested
  budget.
- Seed-to-seed variance was large. Matched seeds and a data generator isolated
  from model initialization were necessary for fair architecture comparisons.
- A quality-diversity population run selected the strongest no-gate reference
  checkpoint. Its behavioral signature used the full channel-0 state with
  additional weight on output cells.

Population schedules based on robust log-loss projection and behavioral
diversity were discussed but not established as a validated canonical method.
They are intentionally absent from the reduced implementation.
