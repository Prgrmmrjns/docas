"""Domain-agnostic synthetic alignment (primary public API)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from docas.intervene import InterventionModel, probe


@dataclass
class AlignResult:
    model_: object
    baseline_: object
    n_synth: int
    n_real: int
    alignment_error: float
    meta: dict = field(default_factory=dict)

    @property
    def model(self):
        return self.model_


class Aligner:
    """Align any fit/predict regressor to a dose–response target τ."""

    def __init__(
        self,
        *,
        train_fn,
        intervention_idx: int,
        target,
        context=None,
        span: float = 1.0,
        mode: str = "channel",
        n_anchors: int = 200,
        n_u: int = 11,
        synth_weight: float = 40.0,
        align_passes: int = 1,
        seed: int = 42,
        u_grid: np.ndarray | None = None,
    ):
        if mode not in {"channel", "add", "replace"}:
            raise ValueError("mode must be 'channel', 'add', or 'replace'")
        self.train_fn = train_fn
        self.ji = int(intervention_idx)
        self.target = target
        self.context = context or (lambda _X, yhat: np.isfinite(yhat))
        self.span = float(span)
        self.mode = mode
        self.n_anchors = int(n_anchors)
        self.n_u = int(n_u)
        self.synth_weight = float(synth_weight)
        self.align_passes = max(1, int(align_passes))
        self.seed = int(seed)
        self.u_grid = np.linspace(0.0, 1.0, 11) if u_grid is None else np.asarray(u_grid, float)
        self.model_ = self.baseline_ = self.result_ = None

    def _baseline_pred(self, model, X):
        return np.asarray(model.predict(np.asarray(X, float)), float).ravel()

    def _synth(self, X, baseline, rng):
        X = np.asarray(X, float)
        yhat = self._baseline_pred(baseline, X)
        mask = np.asarray(self.context(X, yhat), bool).ravel()
        idx = np.where(mask & np.isfinite(yhat))[0]
        if idx.size == 0:
            return np.zeros((0, X.shape[1] + (1 if self.mode == "channel" else 0)), float), np.zeros(0, float)
        if 0 < self.n_anchors < idx.size:
            idx = rng.choice(idx, self.n_anchors, replace=False)
        u = np.linspace(0.0, 1.0, max(2, self.n_u))
        Xs = np.repeat(X[idx], len(u), 0)
        u_use = np.tile(u, len(idx))
        ys = np.asarray(self.target(Xs, u_use, baseline), float).ravel()
        if self.mode == "channel":
            Xs = np.column_stack([Xs, u_use * self.span])
        elif self.mode == "add":
            Xs[:, self.ji] = Xs[:, self.ji] + u_use * self.span
        else:
            Xs[:, self.ji] = u_use * self.span
        return Xs, ys

    def alignment_error(self, model, baseline, X, *, u=None) -> float:
        """RMSE between audit curve and target on context-selected rows."""
        X = np.asarray(X, float)
        u = self.u_grid if u is None else np.asarray(u, float).ravel()
        yhat = self._baseline_pred(baseline, X)
        mask = np.asarray(self.context(X, yhat), bool).ravel() & np.isfinite(yhat)
        if not np.any(mask):
            return float("inf")
        Xa = X[mask]
        mode = "channel" if self.mode == "channel" else self.mode
        pred = probe(model, Xa, intervention_idx=self.ji, u=u, span=self.span, mode=mode)
        # target for each (row, u): build like synth
        want = np.zeros_like(pred)
        for j, uj in enumerate(u):
            uu = np.full(len(Xa), float(uj))
            want[:, j] = np.asarray(self.target(Xa, uu, baseline), float).ravel()
        err = pred - want
        return float("inf") if not np.isfinite(err).all() else float(np.sqrt(np.mean(err ** 2)))

    def fit(self, X, y, *, baseline=None) -> AlignResult:
        X, y = np.asarray(X, float), np.asarray(y, float).ravel()
        rng = np.random.default_rng(self.seed)
        baseline = baseline or self.train_fn(X, y)
        Xs, ys = self._synth(X, baseline, rng)
        if self.mode == "channel":
            Xr = np.column_stack([X, np.zeros(len(X))])
        else:
            Xr = X.copy()
        if len(ys) == 0:
            raw = self.train_fn(Xr, y)
            model = InterventionModel(raw, X.shape[1]) if self.mode == "channel" else raw
        else:
            w = np.concatenate([np.ones(len(Xr)), np.full(len(ys), self.synth_weight)])
            Xt, ys_fit = np.vstack([Xr, Xs]), ys.copy()
            for _ in range(self.align_passes):
                raw = self.train_fn(Xt, np.concatenate([y, ys_fit]), sample_weight=w)
                model = InterventionModel(raw, X.shape[1]) if self.mode == "channel" else raw
                if self.align_passes > 1:
                    ys_fit = ys + (ys - np.asarray(model.predict(Xs), float).ravel())
        result = AlignResult(
            model_=model, baseline_=baseline, n_synth=int(len(ys)), n_real=int(len(X)),
            alignment_error=self.alignment_error(model, baseline, X),
            meta=dict(mode=self.mode, span=self.span, n_u=self.n_u, synth_weight=self.synth_weight),
        )
        self.model_, self.baseline_, self.result_ = model, baseline, result
        return result
