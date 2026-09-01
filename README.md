# ncpu-simplified

`ncpu-simplified` studies whether a very small neural cellular automaton can
perform exact binary addition using sparse, single-cell inputs and outputs.
Every cell applies the same local update rule. The operands are initial
conditions; computation and memory must emerge in the evolving grid state.

The primary inspiration for this project is Iliya Zhechev's
[`ichko/ncpu`](https://github.com/ichko/ncpu). This repository is a fresh,
reduced implementation with independent Git history, built to explore and
extend that central idea. It is also informed by the
[*Emergent Models: Intelligence from Tiny Substrates*](https://arxiv.org/abs/2608.14019)
theoretical framework and by Béna and Faldor's
[*A Path to Universal Neural Cellular Automata*](https://arxiv.org/abs/2505.13058).
The long-term goal is a trainable program channel that selects different
computations while retaining one shared NCA update rule.

> [!WARNING]
> This project is currently documented and tested only on Windows. Other
> operating systems are not supported by the installation instructions, and
> `visual_inference.bat` will not run outside Windows without replacement.

> [!CAUTION]
> This project has been AI-assisted by OpenAI Codex. Although it has been
> tested, bugs or unexpected behaviours may still be present.

## Canonical experiment

The default model follows the strongest compact no-gate setup from the
development experiments:

- two 4-bit operands and a 5-bit sum;
- exact ternary cells: `-1` for zero, `+1` for one, and `0` for background;
- three persistent and fully mutable state channels;
- input implanted only in channel 0 at time zero;
- identity, Sobel-X, and Sobel-Y fixed perception kernels;
- one shared learnable 3x3 kernel initialized as a normalized Laplacian;
- a 57-unit per-cell hidden layer and no residual write gate;
- 70 unsupervised evolution steps followed by supervision on every state from
  step 71 through step 140;
- zero padding, matching the boundary condition used in the experiments.

The default model has 921 trainable parameters. Configuration also supports a
hardcoded Laplacian, `K` learnable shared kernels initialized from the
Laplacian or from a reproducible random distribution, alternative gates,
arbitrary channel counts, stochastic updates, geometry changes, and a frozen
input channel.

## Setup on Windows

The supplied Conda environment targets Python 3.10, PyTorch 2.5.1, and CUDA
12.1. Start PowerShell in the directory where you want the repository, then
clone it and enter its root:

```powershell
git clone https://github.com/BoccheseGiacomo/ncpu-simplified.git
Set-Location .\ncpu-simplified
```

Install from that repository root. These explicit paths do not require
`conda activate` and match a standard Anaconda installation under your Windows
user profile:

```powershell
$CondaExe = "$env:USERPROFILE\anaconda3\Scripts\conda.exe"
$PythonExe = "$env:USERPROFILE\anaconda3\envs\slackenv\python.exe"
& $CondaExe env update -n slackenv -f .\environment.yml
& $PythonExe -m pip install -e ".[dev]"
& $PythonExe -m pytest -q
& $PythonExe -m black --check src tests
& $PythonExe -m flake8 src tests
```

The editable installation is important: the notebook and local server import
the same package tested by `pytest`. Run these commands from the repository
root so that `.` identifies this checkout. If Anaconda is installed elsewhere,
change only `$CondaExe` and `$PythonExe`.

## Notebook workflow

Open [run/run.ipynb](run/run.ipynb) with the `slackenv` kernel. Its four code
sections are:

1. imports, complete experiment settings, a grid and memory report, and the
   fast `validate()` suite;
2. training or exact checkpoint resumption;
3. exhaustive validation at 4, 6, and 8 bits;
4. an embedded RGB animation of one inference.

Checkpoints are written to `checkpoints/`. That directory and all PyTorch
checkpoint extensions are deliberately ignored by Git.

Set `SEEDS` in the first cell to train one or more independent seeds. Each seed
has isolated model and data randomness and writes under `checkpoints/seed_N/`.
The best exhaustive validation checkpoint becomes `checkpoints/best.pt`.

W&B reporting is disabled by default. To enable it, install the tracking extra
and set `WANDB_PROJECT` in the notebook:

```powershell
& "$env:USERPROFILE\anaconda3\envs\slackenv\python.exe" -m pip install -e ".[tracking]"
```

## Browser visualizer

After training has created `checkpoints/best.pt`, run:

```powershell
.\visual_inference.bat
```

The script uses the `slackenv` interpreter directly and opens
`http://127.0.0.1:8000`. Operands, bit width, rollout length, playback, and the
time step can be changed in the page. Channels 0, 1, and 2 map to red, green,
and blue after bounded `tanh` display normalization.

## Source layout

```text
src/ncpu_simplified/
  config.py       experiment configuration and invariants
  layout.py       sparse ternary geometry, encoding, and decoding
  model.py        perception, update rule, gates, and NCA evolution
  training.py     supervised timing, optimization, and checkpoints
  evaluation.py   exact accuracy, MSE, stability, and extrapolation
  validation.py   fast structural and numerical validation
  visualize.py    GIF utilities and the local Flask server
```

Past results and their comparability limits are recorded in
[EXPERIMENTS.md](EXPERIMENTS.md).

## Attribution and licensing

The architecture and experiments developed from the questions explored in
`ichko/ncpu`, and that lineage should remain visible in derived work. The
upstream repository does not currently include a license file. This repository
therefore uses a clean implementation rather than copying its source. No
separate license is granted here unless a license file is added later.
