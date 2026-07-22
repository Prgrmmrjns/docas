"""T1D pipeline: scaled ΔCGM labels and ReplayBG-style fractional τ.

Research scripts use this module. For domain-agnostic use see :class:`docas.Aligner`.
"""
from __future__ import annotations

import numpy as np

from docas.intervene import InterventionModel

_inv = lambda s, a: s.inverse_transform(np.asarray(a, float).reshape(-1, 1)).reshape(np.asarray(a).shape)

F0_REF = 220.0  # manuscript G_ref


def future(model, X, ji, yi, u, span, a_s, dy_s, y_s, additive=True):
    """Predicted future BG under dose grid ``u``. Shape ``(n, |u|)``."""
    X, u = np.asarray(X, float), np.asarray(u, float)
    if len(X) == 0:
        return np.zeros((0, u.size), float)
    if hasattr(model, "predict_intervention"):
        return _inv(y_s, X[:, yi])[:, None] + _inv(dy_s, model.predict_intervention(X, u, span))
    if additive and u.size == 1 and u.flat[0] == 0.0:
        return (_inv(y_s, X[:, yi]) + _inv(dy_s, model.predict(X)))[:, None]
    rows = np.repeat(X, u.size, 0)
    base = np.repeat(_inv(a_s, X[:, ji]), u.size) if additive else 0.0
    rows[:, ji] = a_s.transform((np.tile(u * span, len(X)) + base).reshape(-1, 1)).ravel()
    return _inv(y_s, X[:, yi])[:, None] + _inv(dy_s, model.predict(rows)).reshape(len(X), u.size)


def target_delta_ref(shape_fn, u, *, y_start=0.0, y_end=None):
    """ΔBG vs f0 at ``F0_REF`` from fractional lowering ``R = shape_fn``."""
    u = np.asarray(u, float)
    r = np.asarray(shape_fn(u), float)
    if y_end is None:
        return float(y_start) - F0_REF * r
    r1 = float(np.asarray(shape_fn([1.0]), float).ravel()[0])
    s = np.clip(np.asarray(r, float).ravel() / max(abs(r1), 1e-12), 0.0, 1.0).reshape(u.shape)
    return float(y_start) + (float(y_end) - float(y_start)) * s


def scale_delta_to_f0(delta_ref, f0, f0_ref=F0_REF):
    return np.asarray(delta_ref, float) * (np.asarray(f0, float) / float(f0_ref))


def sample_synthetic(
    X, y, *, target, baseline, ji, yi, span, a_s, dy_s, y_s,
    n_f0=0, n_u=10, synth_weight=40.0, y0_lo=170.0, y0_hi=280.0, additive=True, f0=None, rng=None,
):
    """Audit net: hyperglycaemic f0-band × evenly spaced interventional u."""
    X, y = np.asarray(X, float), np.asarray(y, float).ravel()
    rng = np.random.default_rng() if rng is None else rng
    Xr = np.column_stack([X, np.zeros(len(X))])
    clin_lo, clin_hi = float(y0_lo), float(y0_hi)
    n_y_req = max(0, int(round(float(n_f0))))
    r1 = float(np.asarray(target([1.0]), float).ravel()[0])
    amp = F0_REF * abs(r1)
    meta = dict(
        y0_lo=clin_lo, y0_hi=clin_hi, n_anchors=0, amplitude=amp, r1=r1,
        n_synth=0, n_real=int(len(Xr)), target_y_start=0.0, target_y_end=-amp,
        synth_weight=float(synth_weight), n_f0=n_y_req, n_u=int(round(float(n_u))),
    )
    if f0 is None:
        f0 = future(baseline, X, ji, yi, (0.0,), span, a_s, dy_s, y_s, additive).ravel()
    else:
        f0 = np.asarray(f0, float).ravel()
    band = (f0 > clin_lo) & (f0 < clin_hi) & np.isfinite(f0)
    if int(band.sum()) == 0:
        return Xr, y, np.zeros((0, X.shape[1] + 1), float), np.zeros(0, float), np.zeros(0, float), meta
    idx_b = np.where(band)[0]
    if 0 < n_y_req < idx_b.size:
        idx_b = rng.choice(idx_b, n_y_req, replace=False)
    u_net = np.linspace(0.0, 1.0, max(2, int(round(float(n_u)))))
    g, n_y = int(u_net.size), int(idx_b.size)
    Xs = np.repeat(X[idx_b], g, 0)
    u_use = np.tile(u_net, n_y)
    f0_p = np.repeat(f0[idx_b].astype(float), g)
    r = np.asarray(target(u_use), float).ravel()
    ys = dy_s.transform((f0_p * (1.0 - r) - _inv(y_s, Xs[:, yi])).reshape(-1, 1)).ravel()
    Xs = np.column_stack([Xs, u_use * float(span)])
    meta.update(n_synth=int(len(ys)), n_anchors=n_y, n_u=g)
    return Xr, y, Xs, ys, np.ones(len(ys), float), meta


class DOCAS:
    """Paper-facing aligner for scaled Δ-outcome glucose forecasts.

    Class attributes (``DOCAS.SPAN``, ``N_F0``, …) are the study knobs used by
    ``scripts/``. Prefer :class:`docas.Aligner` for new / non-T1D projects.
    """

    LOAD = False
    SPAN = 10.0
    Y0_LO, Y0_HI = 170.0, 280.0
    N_F0 = 300
    N_U = 21
    SYNTH_WEIGHT = 40.0
    ALIGN_PASSES = 2
    CORRECTION_GAIN = 1.0
    SEED = 42
    TUNE_CTX_N = 80
    U_GRID = np.linspace(0.0, 1.0, 10)
    train_fn = None

    def __init__(
        self, target, *, intervention_idx, outcome_idx, action_scaler, level_scaler, delta_scaler,
        span=None, train_fn=None, additive=True, u_grid=None, seed=None,
        model_kwargs=None, target_name="user_target", intervention_name="additive",
    ):
        C = type(self)
        self.target, self.ji, self.yi = target, int(intervention_idx), int(outcome_idx)
        self.a_s, self.y_s, self.dy_s = action_scaler, level_scaler, delta_scaler
        self.span = float(C.SPAN if span is None else span)
        self.train_fn = train_fn or getattr(C, "train_fn", None)
        self.additive = bool(additive)
        self.u_grid = np.asarray(C.U_GRID if u_grid is None else u_grid, float)
        self.seed = int(C.SEED if seed is None else seed)
        self.y0_lo, self.y0_hi = float(C.Y0_LO), float(C.Y0_HI)
        self.tune_ctx_n = int(C.TUNE_CTX_N)
        self.model_kwargs = dict(model_kwargs or {})
        self.target_name, self.intervention_name = target_name, intervention_name
        self.model_ = self.baseline_ = self.meta_ = None

    def alignment(self, model, baseline, X, *, u=None, y0_lo=None, y0_hi=None, response=None, f0=None):
        X, u = np.asarray(X, float), np.asarray(self.u_grid if u is None else u, float)
        y0_lo = self.y0_lo if y0_lo is None else y0_lo
        y0_hi = self.y0_hi if y0_hi is None else y0_hi
        if not np.isfinite(y0_lo) or not np.isfinite(y0_hi) or y0_hi <= y0_lo:
            y0_lo, y0_hi = self.y0_lo, self.y0_hi
        if response is None:
            response = target_delta_ref(self.target, u)
        response = np.asarray(response, float).ravel()
        if f0 is None:
            f0 = future(baseline, X, self.ji, self.yi, (0.0,), self.span, self.a_s, self.dy_s, self.y_s, self.additive).ravel()
        else:
            f0 = np.asarray(f0, float).ravel()
        m = (f0 > y0_lo) & (f0 < y0_hi) & np.isfinite(f0)
        if not np.any(m):
            m = (f0 > self.y0_lo) & (f0 < self.y0_hi) & np.isfinite(f0)
        if not np.any(m):
            return float("inf")
        scale = (f0[m] / F0_REF)[:, None]
        err = future(model, X[m], self.ji, self.yi, u, self.span, self.a_s, self.dy_s, self.y_s, self.additive)
        err = err - (f0[m][:, None] + response[None, :] * scale)
        return float("inf") if not np.isfinite(err).all() else float(np.sqrt(np.mean(err ** 2)))

    def fit(self, X_train, y_train, *, baseline_model=None, fixed_params=None, u_norm=None):
        if self.train_fn is None:
            raise RuntimeError("DOCAS.train_fn is not set (paper scripts call set_backend first)")
        C = type(self)
        X, y = np.asarray(X_train, float), np.asarray(y_train, float).ravel()
        kw, fp = {**self.model_kwargs}, dict(fixed_params or {})
        ug, span, ji, yi = self.u_grid, self.span, self.ji, self.yi
        u_audit = np.asarray(u_norm, float) if u_norm is not None else (
            np.linspace(0.0, 1.0, int(fp["grid_n"])) if fp.get("grid_n") is not None else ug)
        baseline = baseline_model or self.train_fn(X, y, model_kwargs=kw)
        rng = np.random.default_rng(self.seed)
        delta = target_delta_ref(self.target, u_audit).ravel()
        r1 = float(np.asarray(self.target([1.0]), float).ravel()[0])
        amp = F0_REF * abs(r1)

        Xr, yr, Xs, ys, ws, meta = sample_synthetic(
            X, y, target=self.target, baseline=baseline, ji=ji, yi=yi, span=span,
            a_s=self.a_s, dy_s=self.dy_s, y_s=self.y_s, additive=self.additive,
            y0_lo=self.y0_lo, y0_hi=self.y0_hi, rng=rng,
            n_f0=int(C.N_F0), n_u=int(C.N_U), synth_weight=float(C.SYNTH_WEIGHT),
        )
        hist = {k: meta[k] for k in ("n_synth", "n_real")}
        if len(ys) == 0:
            model = InterventionModel(self.train_fn(Xr, yr, model_kwargs=kw), X.shape[1])
        else:
            w = np.concatenate([np.ones(len(Xr), float), float(C.SYNTH_WEIGHT) * np.asarray(ws, float)])
            Xt = np.vstack([Xr, Xs])
            ys_fit = ys
            for _ in range(max(1, int(C.ALIGN_PASSES))):
                model = InterventionModel(
                    self.train_fn(Xt, np.concatenate([yr, ys_fit]), model_kwargs=kw, sample_weight=w), X.shape[1])
                ys_fit = ys + float(C.CORRECTION_GAIN) * (ys - model.predict(Xs))
            hist["alignment_passes"] = int(C.ALIGN_PASSES)

        rmse_m = lambda m_: float((((_inv(self.dy_s, m_.predict(X)) - _inv(self.dy_s, y)) ** 2).mean()) ** 0.5)
        X_ctx = X if len(X) <= self.tune_ctx_n else X[rng.choice(len(X), self.tune_ctx_n, replace=False)]
        self.model_, self.baseline_, self.meta_ = model, baseline, dict(
            model=model, baseline_model=baseline,
            y0_lo=self.y0_lo, y0_hi=self.y0_hi, n_anchors=meta["n_anchors"],
            drop_fraction=amp, amplitude=amp, r1=r1,
            target=self.target_name, intervention=self.intervention_name,
            grid_points=ug.size, grid_n=u_audit.size, span=span, n_synth=hist["n_synth"],
            target_amplitude=amp, target_y_start=0.0, target_y_end=-amp,
            train_alignment=self.alignment(model, baseline, X_ctx, u=u_audit, response=delta),
            train_rmse_delta=rmse_m(model) - rmse_m(baseline),
            tune_params=dict(
                sampling="f0_insulin_net", r1=r1, target_amplitude=amp,
                n_synth=hist["n_synth"], y0_lo=self.y0_lo, y0_hi=self.y0_hi,
                n_anchors=meta["n_anchors"], target_y_start=0.0, target_y_end=-amp,
            ),
        )
        return self.meta_
