# DOCAS

<p align="center">
  <img src="assets/docas.png" alt="DOCAS — dose–response curve alignment" width="720"/>
</p>

**Dose-response Curve Alignment via Synthetic augmentation** — train any
`fit`/`predict` regressor so its interventional audit curve matches a
user-specified target τ.

```text
observational data  →  baseline model
                    →  pick anchors (context rule)
                    →  label synthetic (x, dose) with τ
                    →  retrain on real + synthetic
                    →  audit curve ≈ τ
```

## Install

```bash
pip install -e ".[examples]"
# later: pip install docas
```

Minimal dependency: `numpy`. Optional: `examples`, `dev`.

## Quick start

```python
from docas import Aligner
from sklearn.ensemble import HistGradientBoostingRegressor

def train(X, y, sample_weight=None):
    m = HistGradientBoostingRegressor(max_depth=3, random_state=0)
    m.fit(X, y, sample_weight=sample_weight)
    return m

def target(X, u, baseline):
    return baseline.predict(X) * (1.0 - 0.3 * u)

result = Aligner(
    train_fn=train,
    intervention_idx=1,
    target=target,
    context=lambda X, yhat: yhat > 0.0,
    span=1.0,
    mode="channel",
).fit(X_train, y_train)

model = result.model_
print(result.alignment_error)
```

```bash
python examples/01_quickstart.py
python examples/02_custom_target.py
python examples/03_sklearn_pipeline.py
```

## Public API

| Symbol | Role |
|--------|------|
| `Aligner` | Domain-agnostic alignment (prefer this) |
| `audit_curve` / `probe` | Counterfactual dose sweeps |
| `InterventionModel` | Extra dose-channel wrapper |
| `interp_target` / `linear_fraction` | Build τ from knots / slope |
| `DOCAS` | Scaled Δ-outcome / glucose research helper |

## Layout

```text
src/docas/     # library
examples/      # demos
```

Manuscript / OhioT1DM study: private repo `DOCAS-manuscript`.

## License

MIT
