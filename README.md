# DOCAS

<p align="center">
  <img src="assets/docas.png" alt="DOCAS — dose–response curve alignment" width="720"/>
</p>

**Dose–response curve alignment via synthetic augmentation**

DOCAS aligns the treatment response of any `fit` / `predict` regressor with a
stated target dose–response curve (DRC). The model, its features, and its loss
stay unchanged. DOCAS only adds synthetic training rows.

## Why

Models trained on observational data learn how treatment was given, not what it
does. In type 1 diabetes, for example, patients bolus insulin when glucose is
high or when they eat, so large doses appear together with high glucose. A
forecaster can therefore predict glucose well on held-out data and still predict
*higher* glucose for more insulin. Ceteris paribus plots (ICE curves and partial
dependence plots) show this visually. The **alignment error** (AE) quantifies it:
the RMSE between each ICE curve and the target DRC, reported next to the usual
RMSE.

Note that the target DRC is the observed change in outcome when a dose is taken,
whereas an ICE curve is the change in the model's prediction when only the dose
input changes. The target DRC is a stated reference, e.g. a population curve from
pharmacology, and not an individual ground truth.

## How it works

```text
observational (X, y)
        │
        ├─ copy real training rows (with replacement)
        ├─ add small noise to chosen context columns   (jitter)
        ├─ replace the treatment with a probe dose     (half observed doses, half uniform)
        ├─ label = observed y + DRC(probe dose) − DRC(observed dose)
        └─ train once on real + synthetic rows
```

- **Budget.** The number of synthetic rows is a multiple of the training size
  (`synth_ratio`).
- **Probe doses.** With probability `emp_frac`, a probe dose is drawn from the
  doses observed in training, so common doses are well represented. Otherwise it
  is drawn uniformly between zero and the maximum dose (`span`), so rare large
  doses are covered too.
- **Labels.** Everything apart from the dose that influenced the observed
  outcome stays in the label, and only the dose effect follows the target DRC.
- **Training.** One model is trained on the union of observed and synthetic
  rows, so the target DRC sets the treatment response while the observed data
  shape the rest of the model. The response is only learned within the range
  covered by the synthetic rows.

## Install

```bash
pip install -e ".[examples]"
```

Core: `numpy`. Examples and tests: `scikit-learn`.

## Quick start

Toy data (not patient data): severity drives both dose and outcome, so a plain
regressor learns the wrong slope. DOCAS pulls the ICE curves toward a lowering
target DRC.

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from docas import Aligner, audit_curve
from docas.datasets import make_synthetic_glucose


def train(X, y, sample_weight=None):
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=80, random_state=0)
    m.fit(X, y, sample_weight=sample_weight)
    return m


# Target DRC in y units: change in outcome at fractional dose u in [0, 1].
def target(u):
    return -20.0 * np.asarray(u, float)


data = make_synthetic_glucose(n=2_000, seed=0)
X, y = data.X, data.y
ji = data.treatment_idx

baseline = train(X, y)
result = Aligner(
    train,
    treatment_idx=ji,
    target=target,
    span=1.0,
    synth_ratio=2.0,
    emp_frac=0.5,
).fit(X, y, baseline=baseline)

u = np.linspace(0, 1, 6)
print("align. error :", round(result.alignment_error, 3))
print("baseline     :", np.round(audit_curve(baseline, X, treatment_idx=ji, u=u), 2))
print("aligned      :", np.round(audit_curve(result.model_, X, treatment_idx=ji, u=u), 2))
```

```bash
python examples/01_quickstart.py
```

## What you pass

| Argument | Meaning |
|--------|---------|
| `X`, `y` | Whatever your model already trains on |
| `treatment_idx` | The treatment column that is set under `do(·)` |
| `target` | Target DRC in `y` units: `target(u)`, `target(X, u)`, or `target(X, u, baseline)` |
| `span` | Physical dose at `u = 1` (the maximum dose) |

Optional: `context(X, y) -> mask` to restrict which rows are copied;
`jitter={col: std}` for noise on copied contexts. A target that uses `X` can
scale the response with the context, e.g. a larger glucose drop at higher
current glucose:

```python
def target(X, u):
    glucose = X[:, 0]
    return -0.15 * glucose * np.asarray(u, float)
```

## Settings

| Setting | Default | Role |
|------|---------|------|
| `synth_ratio` | `1.0` | Synthetic budget: synthetic rows as a multiple of the training size |
| `emp_frac` | `0.5` | Share of probe doses drawn from observed doses (rest uniform on `[0, 1]`) |
| `jitter` | `None` | `{column: std}` Gaussian noise on copied contexts |
| `mode` | `"replace"` | Set the treatment column to `u·span`. `"add"` increments it. `"append"` adds a dose feature. |
| `n_u` | `None` | If set, pair each copied row with this many evenly spaced doses instead of the mix |
| `label` | `"incremental"` | Observed `y` plus the target DRC difference. `"absolute"` uses the target DRC as `y`. |
| `synth_weight` | `1.0` | Sample weight of synthetic rows, passed to `train_fn` only when not 1 |
| `seed` | `42` | Sampling seed |

`train_fn` may ignore `sample_weight` / `n_factual`; extra arguments are dropped.

## Custom target and any sklearn model

```bash
python examples/02_custom_target.py    # target DRC from knots
python examples/03_sklearn_pipeline.py # Ridge factory
```

Knots:

```python
from docas import Aligner, interp_target

drc = interp_target([0.0, 0.4, 1.0], [0.0, -8.0, -25.0])
Aligner(train, treatment_idx=1, target=drc, span=5.0).fit(X, y)
```

## Public API

| Symbol | Role |
|--------|------|
| `Aligner` / `AlignResult` | Train on real + synthetic rows; `alignment_error` |
| `sample_synthetic` | Build the synthetic rows without training |
| `probe` / `centred_ice` / `audit_curve` | ICE curves and partial dependence |
| `interp_target` / `linear_fraction` | Build a target DRC |
| `temporal_split` | Chronological train / validation cut |
| `docas.datasets.make_synthetic_glucose` | Confounded toy data |

The type 1 diabetes study (OhioT1DM, AZT1D, UVa/Padova target DRC, ReplayBG
evaluation) lives in the
[manuscript repository](https://github.com/Prgrmmrjns/DOCAS-manuscript), not in
this package.

## License

MIT
