"""Align a tabular regressor so its ICE curves follow a stated target DRC."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from docas.intervene import AppendModel, apply_dose, centred_ice


def eval_target(target, X, u, baseline=None):
    """Call ``target`` as ``(X, u, baseline)``, ``(X, u)``, or ``(u,)``."""
    X = np.asarray(X, float)
    u = np.asarray(u, float).ravel()
    if u.size == 1:
        u = np.full(len(X), float(u[0]))
    elif u.size != len(X):
        raise ValueError("u must be scalar or one value per row")
    for args in ((X, u, baseline), (X, u), (u,)):
        try:
            out = np.asarray(target(*args), float)
            break
        except TypeError:
            continue
    else:
        raise TypeError("target must accept (X, u, baseline), (X, u), or (u,)")
    out = np.asarray(out, float).ravel()
    if out.size == 1:
        return np.full(len(X), float(out[0]))
    if out.size != len(X):
        raise ValueError("target must return a scalar or one value per row")
    return out


def sample_synthetic(
    X, y, *, treatment_idx, target, span=1.0, mode="replace",
    synth_ratio=1.0, emp_frac=0.5, n_u=None, context=None,
    jitter=None, baseline=None, rng=None, label="incremental",
):
    """Build synthetic ``do(treatment)`` rows labelled by the target DRC.

    Each row copies a real context (optionally with ``jitter``) and replaces the
    treatment with a probe dose. Probe doses are drawn from the observed doses
    with probability ``emp_frac`` and uniformly on ``[0, 1]`` otherwise. Labels
    keep the observed outcome and add the target DRC difference between the
    probe dose and the observed dose, unless ``label="absolute"``.
    """
    X, y = np.asarray(X, float), np.asarray(y, float).ravel()
    rng = np.random.default_rng() if rng is None else rng
    span = float(span)
    mask = np.ones(len(X), bool)
    if context is not None:
        yhat = np.asarray(baseline.predict(X), float).ravel() if baseline is not None else y
        try:
            mask = np.asarray(context(X, yhat), bool).ravel()
        except TypeError:
            try:
                mask = np.asarray(context(X, y), bool).ravel()
            except TypeError:
                mask = np.asarray(context(X), bool).ravel()
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        extra = 1 if mode == "append" else 0
        return np.zeros((0, X.shape[1] + extra), float), np.zeros(0, float), {}

    n = max(1, int(round(float(synth_ratio) * len(y))))
    u_pool = np.clip(X[idx, treatment_idx] / max(span, 1e-12), 0.0, 1.0)
    if n_u is not None:
        n_u = max(2, int(n_u))
        n_a = max(1, int(np.ceil(n / n_u)))
        a = rng.choice(idx, n_a, replace=True)
        grid = np.linspace(0.0, 1.0, n_u)
        i = np.repeat(a, n_u)
        u = np.tile(grid, n_a)[: len(i)]
    else:
        i = rng.choice(idx, n, replace=True)
        emp = rng.random(n) < float(np.clip(emp_frac, 0.0, 1.0))
        u = np.where(emp, rng.choice(u_pool, n), rng.random(n))

    Xs = X[i].copy()
    if jitter:
        for col, sd in dict(jitter).items():
            if float(sd) > 0:
                Xs[:, int(col)] = Xs[:, int(col)] + rng.normal(0.0, float(sd), len(Xs))

    u_obs = np.clip(Xs[:, treatment_idx] / max(span, 1e-12), 0.0, 1.0)
    Xs = apply_dose(Xs, treatment_idx=treatment_idx, u=u, span=span, mode=mode)
    drc_u = eval_target(target, Xs if mode != "append" else X[i], u, baseline)
    drc_o = eval_target(target, X[i], u_obs, baseline)
    if str(label).lower().startswith("abs"):
        ys = drc_u
    else:
        ys = y[i] + drc_u - drc_o
    return Xs, ys, dict(n_synth=int(len(ys)), n_anchors=int(len(np.unique(i))), u=np.asarray(u, float), anchor_idx=np.asarray(i, int))


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
    """Train once on observational rows plus synthetic interventional rows.

    Parameters
    ----------
    train_fn :
        ``(X, y, sample_weight=None, **model_kwargs) -> model`` with ``predict``.
    treatment_idx :
        Column rewritten under ``do(·)``.
    target :
        Target DRC: expected change in outcome at fractional dose ``u`` versus
        no dose, in ``y`` units. Accepted as ``target(X, u, baseline)``, ``target(X, u)``, or ``target(u)``.
    span :
        Physical treatment at fractional dose ``u = 1``.
    mode :
        ``"replace"`` (default; set the column to ``u·span``), ``"add"``, or
        ``"append"`` (extra dose feature; observational rows get 0).
    synth_ratio :
        Synthetic rows as a multiple of the training length.
    emp_frac :
        Share of synthetic doses drawn from observed treatment (rest Uniform).
    n_u :
        If set, pair each anchor with this many evenly spaced doses instead of
        the empiric/Uniform mix.
    context :
        Optional ``(X, y) -> bool mask`` restricting which rows are anchors.
    jitter :
        Optional ``{column: std}`` Gaussian noise on copied contexts.
    label :
        ``"incremental"`` (paper default) or ``"absolute"``.
    """

    def __init__(
        self,
        train_fn,
        *,
        treatment_idx: int | None = None,
        intervention_idx: int | None = None,
        target,
        span: float = 1.0,
        mode: str = "replace",
        synth_ratio: float = 1.0,
        emp_frac: float = 0.5,
        n_u: int | None = None,
        context=None,
        jitter: dict | None = None,
        label: str = "incremental",
        model_kwargs: dict | None = None,
        seed: int = 42,
        synth_weight: float = 1.0,
        u_grid: np.ndarray | None = None,
    ):
        if treatment_idx is None and intervention_idx is None:
            raise ValueError("pass treatment_idx")
        if treatment_idx is not None and intervention_idx is not None:
            if int(treatment_idx) != int(intervention_idx):
                raise ValueError("treatment_idx and intervention_idx disagree")
        if mode not in {"replace", "add", "append"}:
            raise ValueError("mode must be 'replace', 'add', or 'append'")
        self.train_fn = train_fn
        self.treatment_idx = int(treatment_idx if treatment_idx is not None else intervention_idx)
        self.intervention_idx = self.treatment_idx
        self.target = target
        self.span = float(span)
        self.mode = mode
        self.synth_ratio = float(synth_ratio)
        self.emp_frac = float(emp_frac)
        self.n_u = None if n_u is None else int(n_u)
        self.context = context
        self.jitter = None if jitter is None else dict(jitter)
        self.label = str(label)
        self.model_kwargs = dict(model_kwargs or {})
        self.seed = int(seed)
        self.synth_weight = float(synth_weight)
        self.u_grid = np.linspace(0.0, 1.0, 21) if u_grid is None else np.asarray(u_grid, float)
        self.model_ = self.baseline_ = self.result_ = None

    def with_params(self, **kw):
        cfg = dict(
            train_fn=self.train_fn, treatment_idx=self.treatment_idx, target=self.target,
            span=self.span, mode=self.mode, synth_ratio=self.synth_ratio,
            emp_frac=self.emp_frac, n_u=self.n_u, context=self.context,
            jitter=self.jitter, label=self.label, model_kwargs=self.model_kwargs,
            seed=self.seed, synth_weight=self.synth_weight, u_grid=self.u_grid,
        )
        cfg.update(kw)
        return type(self)(**cfg)

    def _train(self, X, y, *, sample_weight=None, n_factual=None):
        kw = dict(self.model_kwargs)
        attempts = [
            dict(sample_weight=sample_weight, n_factual=n_factual, **kw),
            dict(sample_weight=sample_weight, **kw),
            dict(n_factual=n_factual, **kw),
            dict(**kw),
        ]
        for args in attempts:
            args = {k: v for k, v in args.items() if v is not None}
            try:
                return self.train_fn(X, y, **args) if args else self.train_fn(X, y)
            except TypeError:
                continue
        return self.train_fn(X, y)

    def alignment_error(self, model, X, *, u=None, baseline=None) -> float:
        """Alignment error: RMSE between the centred ICE curves and the target DRC."""
        X = np.asarray(X, float)
        u = self.u_grid if u is None else np.asarray(u, float).ravel()
        yhat = np.asarray((baseline if baseline is not None else model).predict(X), float).ravel()
        if self.context is not None:
            try:
                mask = np.asarray(self.context(X, yhat), bool).ravel()
            except TypeError:
                mask = np.asarray(self.context(X), bool).ravel()
            if np.any(mask):
                X = X[mask]
        ice = centred_ice(model, X, treatment_idx=self.treatment_idx, u=u, span=self.span, mode=self.mode)
        want = np.column_stack([
            eval_target(self.target, X, uj, baseline if baseline is not None else self.baseline_)
            for uj in u
        ])
        want0 = eval_target(self.target, X, 0.0, baseline if baseline is not None else self.baseline_)
        err = ice - (want - want0[:, None])
        return float("inf") if not np.isfinite(err).all() else float(np.sqrt(np.mean(err ** 2)))

    def fit(self, X, y, *, baseline=None) -> AlignResult:
        X, y = np.asarray(X, float), np.asarray(y, float).ravel()
        rng = np.random.default_rng(self.seed)
        baseline = baseline or self._train(X, y, n_factual=len(X))
        Xs, ys, meta = sample_synthetic(
            X, y, treatment_idx=self.treatment_idx, target=self.target,
            span=self.span, mode=self.mode, synth_ratio=self.synth_ratio,
            emp_frac=self.emp_frac, n_u=self.n_u, context=self.context,
            jitter=self.jitter, baseline=baseline, rng=rng, label=self.label,
        )
        if self.mode == "append":
            Xr = np.column_stack([X, np.zeros(len(X))])
        else:
            Xr = X
        if len(ys) == 0:
            model = self._train(Xr, y, n_factual=len(X))
        elif abs(self.synth_weight - 1.0) < 1e-12:
            model = self._train(np.vstack([Xr, Xs]), np.concatenate([y, ys]), n_factual=len(X))
        else:
            w = np.concatenate([np.ones(len(Xr)), np.full(len(ys), self.synth_weight)])
            model = self._train(
                np.vstack([Xr, Xs]), np.concatenate([y, ys]),
                sample_weight=w, n_factual=len(X),
            )
        if self.mode == "append":
            model = AppendModel(model, X.shape[1])

        ae = self.alignment_error(model, X, baseline=baseline)
        result = AlignResult(
            model_=model, baseline_=baseline, n_synth=int(len(ys)), n_real=int(len(X)),
            alignment_error=ae,
            meta=dict(
                mode=self.mode, span=self.span, synth_ratio=self.synth_ratio,
                emp_frac=self.emp_frac, n_u=self.n_u, label=self.label,
                treatment_idx=self.treatment_idx, **{k: meta[k] for k in ("n_anchors",) if k in meta},
            ),
        )
        self.model_, self.baseline_, self.result_ = model, baseline, result
        return result


def temporal_split(X, y, *, val_frac: float = 0.5):
    """Chronological cut: first ``1 − val_frac`` for fitting."""
    X, y = np.asarray(X, float), np.asarray(y, float).ravel()
    n = max(1, len(X) - max(1, int(round(len(X) * float(val_frac)))))
    return X[:n], y[:n], X[n:], y[n:]
