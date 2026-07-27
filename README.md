# DOCAS

<p align="center">
  <img src="assets/docas.png" alt="DOCAS — dose–response curve alignment" width="720"/>
</p>

**Dose-response Curve Alignment via Synthetic augmentation**

Train any `fit` / `predict` regressor so its *interventional* dose–response
matches a target curve τ you specify — without rewriting the model.

```text
observational data  →  baseline model
                    →  pick contexts
                    →  label synthetic (x, dose) rows with τ
                    →  train once on real + synthetic
                    →  audit curve ≈ τ
```

Works for any tabular regressor. The treatment can be a column you edit, or an
extra dose feature appended at train / probe time.

## Install

```bash
pip install -e ".[examples]"
```

Core dependency: `numpy`. Examples need `scikit-learn`.

## Quick start

We ship a **synthetic** confounded panel (not real patient data) so you can run
this without external downloads. Severity drives both high treatment and a high
outcome, so a plain regressor learns the wrong effect; DOCAS pulls the audit
curve toward τ.

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from docas import Aligner, audit_curve
from docas.datasets import make_synthetic_glucose

data = make_synthetic_glucose(n=2_000, seed=0)
X, y = data.X, data.y


def train(X, y, sample_weight=None):
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=80, random_state=0)
    m.fit(X, y, sample_weight=sample_weight)
    return m


# Desired response: lower the baseline prediction as dose u goes 0 → 1.
def target(X_rows, u, baseline):
    return baseline.predict(X_rows) - 20.0 * u


baseline = train(X, y)
result = Aligner(
    train_fn=train,
    treatment_idx=data.treatment_idx,
    target=target,
    context=lambda _X, yhat: yhat > np.median(yhat),
    span=1.0,
    mode="append",
    n_anchors=300,
    n_u=11,
    synth_weight=25.0,
).fit(X, y, baseline=baseline)

u = np.linspace(0, 1, 6)
hi = X[baseline.predict(X) > np.median(baseline.predict(X))]
print("align. error :", round(result.alignment_error, 3))
print("baseline     :", np.round(
    audit_curve(baseline, hi, intervention_idx=data.treatment_idx, u=u, span=1.0, mode="add"), 2))
print("aligned      :", np.round(
    audit_curve(result.model_, hi, intervention_idx=data.treatment_idx, u=u, span=1.0, mode="append"), 2))
```

```bash
python examples/01_quickstart.py
```

## How to use it on your task

1. **Features / labels** — any `X`, `y` your model already accepts.
2. **Treatment column** — `treatment_idx` (which feature you intervene on).
3. **Target τ** — a callable `(X_rows, u, baseline) -> y` for dose `u ∈ [0, 1]`.
4. **Context rule** (optional) — which rows enter the synthetic grid
   (`context(X, yhat) -> bool mask`).
5. **Mode**
   - `"append"` — train with an extra dose column (recommended default)
   - `"add"` / `"replace"` — edit the treatment column in place

That is the whole core API. Domains differ only in how you build `X`, `y`, and τ.

## Useful knobs

| Knob | Role |
|------|------|
| `n_anchors` | How many contexts enter the synthetic grid |
| `n_u` | Dose-grid density |
| `synth_weight` | Weight of synthetic vs real rows |
| `span` | Physical dose at `u = 1` |
| `model_kwargs` | Forwarded into `train_fn` |
| `with_params(**kw)` | Copy with updated knobs (for tuning loops) |

```python
from docas import temporal_split, objective_sum

Xtr, ytr, Xva, yva = temporal_split(X, y, val_frac=0.3)
trial = Aligner(train_fn=train, treatment_idx=1, target=target).with_params(n_u=21)
res = trial.fit(Xtr, ytr)
```

## Public API

| Symbol | Role |
|--------|------|
| `Aligner` / `AlignResult` | Core aligner |
| `audit_curve` / `probe` | Counterfactual dose sweeps |
| `InterventionModel` | Extra dose-column wrapper |
| `interp_target` / `linear_fraction` | Build τ from knots / slope |
| `temporal_split` / `scale_target` / `objective_sum` | Tuning helpers |
| `docas.datasets.make_synthetic_glucose` | Toy confounded demo data |

Optional T1D research helpers (`DOCAS`, `future`, …) live in `docas.t1d` and
remain importable from the top level for compatibility.

## Examples

```bash
python examples/01_quickstart.py       # synthetic glucose + Aligner
python examples/02_custom_target.py    # knot-interpolated τ
python examples/03_sklearn_pipeline.py # any sklearn regressor factory
```

## License

MIT
