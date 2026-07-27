"""Domain-agnostic dose–response alignment (primary public API)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from docas.intervene import InterventionModel, probe


@dataclass
class AlignResult:
    """Output of :meth:`Aligner.fit`."""

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
    """Align any ``fit``/``predict`` regressor to a dose–response target τ.

    You supply three things:

    1. ``train_fn(X, y, sample_weight=None, **model_kwargs) -> model``
    2. which column is the treatment (``treatment_idx`` / ``intervention_idx``)
    3. ``target(X_rows, u, baseline) -> y`` — desired labels under dose ``u∈[0,1]``

    ``fit`` trains a baseline (unless given), synthesises interventional rows
    labelled by τ, then trains once on real + synthetic data.
    """

    def __init__(
        self,
        *,
        train_fn,
        target,
        treatment_idx: int | None = None,
        intervention_idx: int | None = None,
        context=None,
        span: float = 1.0,
        mode: str = "append",
        n_anchors: int = 200,
        n_u: int = 11,
        synth_weight: float = 40.0,
        model_kwargs: dict | None = None,
        seed: int = 42,
        u_grid: np.ndarray | None = None,
    ):
        if treatment_idx is None and intervention_idx is None:
            raise ValueError("pass treatment_idx (or intervention_idx)")
        if treatment_idx is not None and intervention_idx is not None:
            if int(treatment_idx) != int(intervention_idx):
                raise ValueError("treatment_idx and intervention_idx disagree")
        ji = int(treatment_idx if treatment_idx is not None else intervention_idx)
        if mode == "channel":
            mode = "append"  # compat alias
        if mode not in {"append", "add", "replace"}:
            raise ValueError("mode must be 'append', 'add', or 'replace'")
        self.train_fn = train_fn
        self.ji = ji
        self.treatment_idx = ji
        self.intervention_idx = ji
        self.target = target
        self.context = context or (lambda _X, yhat: np.isfinite(yhat))
        self.span = float(span)
        self.mode = mode
        self.n_anchors = int(n_anchors)
        self.n_u = int(n_u)
        self.synth_weight = float(synth_weight)
        self.model_kwargs = dict(model_kwargs or {})
        self.seed = int(seed)
        self.u_grid = np.linspace(0.0, 1.0, 11) if u_grid is None else np.asarray(u_grid, float)
        self.model_ = self.baseline_ = self.result_ = None

    def _train(self, X, y, *, sample_weight=None):
        kw = dict(self.model_kwargs)
        try:
            return self.train_fn(X, y, sample_weight=sample_weight, **kw)
        except TypeError:
            if sample_weight is None:
                return self.train_fn(X, y, **kw) if kw else self.train_fn(X, y)
            return self.train_fn(X, y, sample_weight=sample_weight)

    def _baseline_pred(self, model, X):
        return np.asarray(model.predict(np.asarray(X, float)), float).ravel()

    def with_params(self, **kw):
        """Return a copy with updated knobs (handy for hyperparameter search)."""
        cfg = dict(
            train_fn=self.train_fn,
            treatment_idx=self.ji,
            target=self.target,
            context=self.context,
            span=self.span,
            mode=self.mode,
            n_anchors=self.n_anchors,
            n_u=self.n_u,
            synth_weight=self.synth_weight,
            model_kwargs=self.model_kwargs,
            seed=self.seed,
            u_grid=self.u_grid,
        )
        cfg.update(kw)
        return type(self)(**cfg)

    def _synth(self, X, baseline, rng):
        X = np.asarray(X, float)
        yhat = self._baseline_pred(baseline, X)
        mask = np.asarray(self.context(X, yhat), bool).ravel()
        idx = np.where(mask & np.isfinite(yhat))[0]
        if idx.size == 0:
            return np.zeros((0, X.shape[1] + (1 if self.mode == "append" else 0)), float), np.zeros(0, float)
        if 0 < self.n_anchors < idx.size:
            idx = rng.choice(idx, self.n_anchors, replace=False)
        u = np.linspace(0.0, 1.0, max(2, self.n_u))
        Xs = np.repeat(X[idx], len(u), 0)
        u_use = np.tile(u, len(idx))
        ys = np.asarray(self.target(Xs, u_use, baseline), float).ravel()
        if self.mode == "append":
            Xs = np.column_stack([Xs, u_use * self.span])
        elif self.mode == "add":
            Xs[:, self.ji] = Xs[:, self.ji] + u_use * self.span
        else:
            Xs[:, self.ji] = u_use * self.span
        return Xs, ys

    def alignment_error(self, model, baseline, X, *, u=None) -> float:
        """RMSE between the audit curve and target on context-selected rows."""
        X = np.asarray(X, float)
        u = self.u_grid if u is None else np.asarray(u, float).ravel()
        yhat = self._baseline_pred(baseline, X)
        mask = np.asarray(self.context(X, yhat), bool).ravel() & np.isfinite(yhat)
        if not np.any(mask):
            return float("inf")
        Xa = X[mask]
        pred = probe(model, Xa, intervention_idx=self.ji, u=u, span=self.span, mode=self.mode)
        want = np.zeros_like(pred)
        for j, uj in enumerate(u):
            uu = np.full(len(Xa), float(uj))
            want[:, j] = np.asarray(self.target(Xa, uu, baseline), float).ravel()
        err = pred - want
        return float("inf") if not np.isfinite(err).all() else float(np.sqrt(np.mean(err ** 2)))

    def fit(self, X, y, *, baseline=None) -> AlignResult:
        X, y = np.asarray(X, float), np.asarray(y, float).ravel()
        rng = np.random.default_rng(self.seed)
        baseline = baseline or self._train(X, y)
        Xs, ys = self._synth(X, baseline, rng)
        if self.mode == "append":
            Xr = np.column_stack([X, np.zeros(len(X))])
        else:
            Xr = X.copy()
        if len(ys) == 0:
            raw = self._train(Xr, y)
            model = InterventionModel(raw, X.shape[1]) if self.mode == "append" else raw
        else:
            w = np.concatenate([np.ones(len(Xr)), np.full(len(ys), self.synth_weight)])
            Xt = np.vstack([Xr, Xs])
            raw = self._train(Xt, np.concatenate([y, ys]), sample_weight=w)
            model = InterventionModel(raw, X.shape[1]) if self.mode == "append" else raw
        result = AlignResult(
            model_=model,
            baseline_=baseline,
            n_synth=int(len(ys)),
            n_real=int(len(X)),
            alignment_error=self.alignment_error(model, baseline, X),
            meta=dict(
                mode=self.mode,
                span=self.span,
                n_u=self.n_u,
                n_anchors=self.n_anchors,
                synth_weight=self.synth_weight,
                treatment_idx=self.ji,
                model_kwargs=dict(self.model_kwargs),
            ),
        )
        self.model_, self.baseline_, self.result_ = model, baseline, result
        return result
