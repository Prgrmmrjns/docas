"""Optional T1D helper (scaled Δ-outcome labels + fractional τ).

Prefer :class:`docas.Aligner` for new work. This module keeps the manuscript /
research pipeline API (``DOCAS``, ``future``, ``sample_synthetic``, …).
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


def _perturb_contexts(
    X, idx_b, n, *, rng, baseline, ji, yi, span, a_s, dy_s, y_s, additive,
    lo, hi, band, ctx_jitter, ctx_cho_jitter, ctx_ins_jitter, ctx_resample,
    f0=None, recompute=True, tries=16,
):
    """Draw ``n`` heterogeneous contexts; recompute ``f0`` when requested.

    With ``ctx_resample``, every feature (CGM, CHO, observational insulin) is
    redrawn independently from its train marginal, then Gaussian-jittered.
    Labels stay self-consistent because ``f0`` is re-evaluated at the new
    context and the clinical band is re-applied.
    """
    partner = [c for c in range(X.shape[1]) if c not in (ji, yi)]
    ctx, f0s, need = [], [], int(n)
    for _ in range(tries):
        if need <= 0:
            break
        k = int(need * 2.0) + 256
        a = rng.choice(idx_b, k, replace=True)
        C = X[a].copy()
        if ctx_resample:
            C[:, yi] = rng.choice(X[:, yi], k, replace=True)
            C[:, ji] = rng.choice(X[:, ji], k, replace=True)
            for c in partner:
                C[:, c] = rng.choice(X[:, c], k, replace=True)
        if ctx_jitter > 0:
            C[:, yi] = np.clip(C[:, yi] + rng.normal(0.0, ctx_jitter, k), lo[yi], hi[yi])
        if ctx_cho_jitter > 0:
            for c in partner:
                C[:, c] = np.clip(C[:, c] + rng.normal(0.0, ctx_cho_jitter, k), lo[c], hi[c])
        if ctx_ins_jitter > 0:
            C[:, ji] = np.clip(C[:, ji] + rng.normal(0.0, ctx_ins_jitter, k), lo[ji], hi[ji])
        if recompute:
            f = future(baseline, C, ji, yi, (0.0,), span, a_s, dy_s, y_s, additive).ravel()
            keep = np.isfinite(f) & (f > band[0]) & (f < band[1])
            C, f = C[keep], f[keep]
        else:
            f = np.asarray(f0, float)[a]
        ctx.append(C[:need])
        f0s.append(f[:need])
        need -= len(C[:need])
    return np.vstack(ctx), np.concatenate(f0s)


def sample_synthetic(
    X, y, *, target, baseline, ji, yi, span, a_s, dy_s, y_s,
    n_f0=0, n_u=10, synth_weight=1.0, y0_lo=170.0, y0_hi=280.0, additive=True, f0=None, rng=None,
    u_mode="density", u_jitter_u=0.5, u_uniform_frac=0.05,
    ctx_jitter=0.12, ctx_cho_jitter=0.0, ctx_ins_jitter=0.0,
    ctx_resample=True, ctx_recompute_target=True,
):
    """Hyperglycaemic anchors with interventional doses labeled by ``target``.

    ``u_mode``:
      - ``"grid"``: Cartesian product of anchors × evenly spaced ``u`` (legacy).
      - ``"density"``: bootstrap observational insulin (+ light jitter); a small
        uniform fraction covers only the sparse high-IOB tail (≥ P75).
      - ``"uniform"``: continuous ``U(0, 1)`` doses (no density match).

    Partner contexts are diversified via independent marginal resampling plus
    per-feature jitter (see :func:`_perturb_contexts`).
    """
    X, y = np.asarray(X, float), np.asarray(y, float).ravel()
    rng = np.random.default_rng() if rng is None else rng
    Xr = np.column_stack([X, np.zeros(len(X))])
    clin_lo, clin_hi = float(y0_lo), float(y0_hi)
    n_y_req = max(0, int(round(float(n_f0))))
    n_u = max(2, int(round(float(n_u))))
    mode = str(u_mode or "density").lower()
    r1 = float(np.asarray(target([1.0]), float).ravel()[0])
    amp = F0_REF * abs(r1)
    ctx_jitter = max(0.0, float(ctx_jitter))
    ctx_cho_jitter = max(0.0, float(ctx_cho_jitter))
    ctx_ins_jitter = max(0.0, float(ctx_ins_jitter))
    ctx_resample = bool(ctx_resample)
    meta = dict(
        y0_lo=clin_lo, y0_hi=clin_hi, n_anchors=0, amplitude=amp, r1=r1,
        n_synth=0, n_real=int(len(Xr)), target_y_start=0.0, target_y_end=-amp,
        synth_weight=float(synth_weight), n_f0=n_y_req, n_u=n_u,
        u_mode=mode, u_jitter_u=float(u_jitter_u), u_uniform_frac=float(u_uniform_frac),
        ctx_jitter=ctx_jitter, ctx_cho_jitter=ctx_cho_jitter, ctx_ins_jitter=ctx_ins_jitter,
        ctx_resample=ctx_resample, ctx_recompute_target=bool(ctx_recompute_target),
    )
    if f0 is None:
        f0 = future(baseline, X, ji, yi, (0.0,), span, a_s, dy_s, y_s, additive).ravel()
    else:
        f0 = np.asarray(f0, float).ravel()
    band = (f0 > clin_lo) & (f0 < clin_hi) & np.isfinite(f0)
    if int(band.sum()) == 0:
        return Xr, y, np.zeros((0, X.shape[1] + 1), float), np.zeros(0, float), np.zeros(0, float), meta
    idx_b = np.where(band)[0]
    span = float(span)
    heterogeneous = ctx_jitter > 0 or ctx_cho_jitter > 0 or ctx_ins_jitter > 0 or ctx_resample
    perturb = lambda n: _perturb_contexts(
        X, idx_b, n, rng=rng, baseline=baseline, ji=ji, yi=yi, span=span,
        a_s=a_s, dy_s=dy_s, y_s=y_s, additive=additive,
        lo=X.min(0), hi=X.max(0), band=(clin_lo, clin_hi),
        ctx_jitter=ctx_jitter, ctx_cho_jitter=ctx_cho_jitter, ctx_ins_jitter=ctx_ins_jitter,
        ctx_resample=ctx_resample, f0=f0, recompute=bool(ctx_recompute_target),
    )

    if mode == "grid":
        if 0 < n_y_req < idx_b.size:
            idx_b = rng.choice(idx_b, n_y_req, replace=False)
        u_net = np.linspace(0.0, 1.0, n_u)
        g, n_y = int(u_net.size), int(idx_b.size)
        a_idx = np.repeat(idx_b, g)
        u_use = np.tile(u_net, n_y)
        if heterogeneous:
            ctx, f0_a = perturb(n_y)
            n_y = int(len(ctx))
            Xs, f0_p = np.repeat(ctx, g, 0), np.repeat(f0_a, g)
            u_use = np.tile(u_net, n_y)
    else:
        n_synth = max(1, n_y_req * n_u)
        a_idx = rng.choice(idx_b, n_synth, replace=True)
        if mode == "uniform":
            u_use = rng.uniform(0.0, 1.0, n_synth)
        else:
            # Density: bootstrap observational insulin; light jitter; optional thin
            # uniform mass only on the sparse high-IOB tail (keeps low-IOB dense).
            ins_obs = np.clip(_inv(a_s, X[:, ji]), 0.0, span)
            doses = rng.choice(ins_obs, n_synth, replace=True)
            jitter = max(0.0, float(u_jitter_u))
            if jitter > 0:
                doses = doses + rng.normal(0.0, jitter, n_synth)
            doses = np.clip(doses, 0.0, span)
            frac = float(np.clip(u_uniform_frac, 0.0, 1.0))
            if frac > 0 and ins_obs.size:
                m = rng.random(n_synth) < frac
                hi = float(np.quantile(ins_obs, 0.75))
                lo_tail = min(hi, span)
                if span > lo_tail + 1e-9:
                    doses[m] = rng.uniform(lo_tail, span, int(m.sum()))
                else:
                    doses[m] = rng.uniform(0.0, span, int(m.sum()))
            u_use = doses / span
        n_y = int(len(np.unique(a_idx)))
        if heterogeneous:
            Xs, f0_p = perturb(n_synth)
            u_use, n_y = u_use[:len(Xs)], int(len(Xs))

    if not heterogeneous:
        Xs = X[a_idx].copy()
        f0_p = f0[a_idx].astype(float)
    r = np.asarray(target(u_use), float).ravel()
    ys = dy_s.transform((f0_p * (1.0 - r) - _inv(y_s, Xs[:, yi])).reshape(-1, 1)).ravel()
    Xs = np.column_stack([Xs, u_use * span])
    meta.update(n_synth=int(len(ys)), n_anchors=n_y, n_u=n_u)
    return Xr, y, Xs, ys, np.ones(len(ys), float), meta


class DOCAS:
    """T1D research aligner (scaled Δ-outcome + fractional τ).

    Prefer :class:`docas.Aligner` unless you need this scaled-Δ pipeline.
    Class attributes (``SPAN``, ``N_F0``, …) are study knobs.
    """

    LOAD = False
    SPAN = 10.0
    Y0_LO, Y0_HI = 170.0, 280.0
    N_F0 = 300
    N_U = 21
    # If set (>0), n_f0 = round(SYNTH_TO_REAL * n_train / N_U) so synth≈ratio·real.
    SYNTH_TO_REAL = None
    SYNTH_WEIGHT = 1.0  # equal weight; compensate with SYNTH_TO_REAL
    # Synthetic dose sampling: "density" (obs insulin + jitter), "uniform", or "grid".
    U_MODE = "density"
    U_JITTER_U = 0.5
    U_UNIFORM_FRAC = 0.05
    # Partner-context heterogeneity: independent marginals + light CGM jitter.
    CTX_JITTER = 0.12         # CGM (scaled); large values hurt alignment
    CTX_CHO_JITTER = 0.0
    CTX_INS_JITTER = 0.0      # observational insulin feature
    CTX_RESAMPLE = True
    CTX_RECOMPUTE_TAU = True
    SEED = 42
    TUNE_CTX_N = 80
    U_GRID = np.linspace(0.0, 1.0, 10)
    train_fn = None

    @classmethod
    def synth_counts(cls, n_train: int, *, n_f0=None, n_u=None) -> tuple[int, int]:
        """Resolve (n_f0, n_u). Relative density wins when SYNTH_TO_REAL is set."""
        nu = int(cls.N_U if n_u is None else n_u)
        nu = max(2, nu)
        ratio = getattr(cls, "SYNTH_TO_REAL", None)
        if ratio is not None and float(ratio) > 0:
            nf0 = max(1, int(round(float(ratio) * int(n_train) / nu)))
        else:
            nf0 = int(cls.N_F0 if n_f0 is None else n_f0)
        return nf0, nu

    def __init__(
        self, target, *, intervention_idx, outcome_idx, action_scaler, level_scaler, delta_scaler,
        span=None, train_fn=None, additive=True, u_grid=None, seed=None,
        n_f0=None, n_u=None, synth_weight=None, amp_scale=1.0,
        model_kwargs=None, target_name="user_target", intervention_name="additive",
    ):
        C = type(self)
        self._target_fn = target
        self.amp_scale = float(amp_scale)
        self.target = lambda u, _t=target, _a=self.amp_scale: _a * np.asarray(_t(u), float)
        self.ji, self.yi = int(intervention_idx), int(outcome_idx)
        self.a_s, self.y_s, self.dy_s = action_scaler, level_scaler, delta_scaler
        self.span = float(C.SPAN if span is None else span)
        self.train_fn = train_fn or getattr(C, "train_fn", None)
        self.additive = bool(additive)
        self.u_grid = np.asarray(C.U_GRID if u_grid is None else u_grid, float)
        self.seed = int(C.SEED if seed is None else seed)
        self.y0_lo, self.y0_hi = float(C.Y0_LO), float(C.Y0_HI)
        self.tune_ctx_n = int(C.TUNE_CTX_N)
        self.n_f0 = int(C.N_F0 if n_f0 is None else n_f0)
        self.n_u = int(C.N_U if n_u is None else n_u)
        self.synth_weight = float(C.SYNTH_WEIGHT if synth_weight is None else synth_weight)
        self.model_kwargs = dict(model_kwargs or {})
        self.target_name, self.intervention_name = target_name, intervention_name
        self.model_ = self.baseline_ = self.meta_ = None

    def with_params(self, **kw):
        """Copy with updated target/synth/model knobs (for Optuna trials)."""
        cfg = dict(
            target=self._target_fn, intervention_idx=self.ji, outcome_idx=self.yi,
            action_scaler=self.a_s, level_scaler=self.y_s, delta_scaler=self.dy_s,
            span=self.span, train_fn=self.train_fn, additive=self.additive,
            u_grid=self.u_grid, seed=self.seed, n_f0=self.n_f0, n_u=self.n_u,
            synth_weight=self.synth_weight, amp_scale=self.amp_scale,
            model_kwargs=self.model_kwargs, target_name=self.target_name,
            intervention_name=self.intervention_name,
        )
        cfg.update(kw)
        return type(self)(**cfg)

    def alignment(self, model, baseline, X, *, u=None, y0_lo=None, y0_hi=None, response=None, f0=None,
                  dose_weights=None):
        """Density-weighted RMSE of per-row audit curves vs τ.

        For each hyperglycaemic context in ``X``, probes the insulin grid ``u``,
        compares to ``response`` (scaled by f0/F0_REF), and aggregates with
        ``dose_weights`` over the dose axis (train-IOB density). Equal weights
        if ``dose_weights`` is omitted.
        """
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
        if not np.isfinite(err).all():
            return float("inf")
        w = np.ones(err.shape[1], float) if dose_weights is None else np.asarray(dose_weights, float).ravel()
        if w.size != err.shape[1] or not np.isfinite(w).all() or float(w.sum()) <= 0:
            w = np.ones(err.shape[1], float)
        w = w / w.sum()
        # Per-context density-weighted curve RMSE, then mean over contexts → overall RMSE.
        return float(np.sqrt(np.mean(np.sum((err ** 2) * w[None, :], axis=1))))

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

        n_f0, n_u = C.synth_counts(len(X), n_f0=self.n_f0, n_u=self.n_u)
        self.n_f0, self.n_u = n_f0, n_u
        Xr, yr, Xs, ys, ws, meta = sample_synthetic(
            X, y, target=self.target, baseline=baseline, ji=ji, yi=yi, span=span,
            a_s=self.a_s, dy_s=self.dy_s, y_s=self.y_s, additive=self.additive,
            y0_lo=self.y0_lo, y0_hi=self.y0_hi, rng=rng,
            n_f0=n_f0, n_u=n_u, synth_weight=self.synth_weight,
            u_mode=getattr(C, "U_MODE", "density"),
            u_jitter_u=getattr(C, "U_JITTER_U", 2.5),
            u_uniform_frac=getattr(C, "U_UNIFORM_FRAC", 0.35),
            ctx_jitter=getattr(C, "CTX_JITTER", 0.12),
            ctx_cho_jitter=getattr(C, "CTX_CHO_JITTER", 0.0),
            ctx_ins_jitter=getattr(C, "CTX_INS_JITTER", 0.0),
            ctx_resample=getattr(C, "CTX_RESAMPLE", True),
            ctx_recompute_target=getattr(C, "CTX_RECOMPUTE_TAU", True),
        )
        hist = {k: meta[k] for k in ("n_synth", "n_real")}
        if len(ys) == 0:
            model = InterventionModel(self.train_fn(Xr, yr, model_kwargs=kw), X.shape[1])
        else:
            Xt = np.vstack([Xr, Xs])
            yt = np.concatenate([yr, ys])
            if abs(self.synth_weight - 1.0) < 1e-12:
                raw = self.train_fn(Xt, yt, model_kwargs=kw)
            else:
                w = np.concatenate([np.ones(len(Xr), float), self.synth_weight * np.asarray(ws, float)])
                raw = self.train_fn(Xt, yt, model_kwargs=kw, sample_weight=w)
            model = InterventionModel(raw, X.shape[1])

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
                amp_scale=self.amp_scale, n_f0=self.n_f0, n_u=self.n_u,
                synth_weight=self.synth_weight,
                synth_to_real=getattr(C, "SYNTH_TO_REAL", None),
                u_mode=meta.get("u_mode", getattr(C, "U_MODE", "density")),
                u_jitter_u=meta.get("u_jitter_u", getattr(C, "U_JITTER_U", 2.5)),
                u_uniform_frac=meta.get("u_uniform_frac", getattr(C, "U_UNIFORM_FRAC", 0.35)),
                ctx_jitter=meta.get("ctx_jitter", getattr(C, "CTX_JITTER", 0.12)),
                ctx_cho_jitter=meta.get("ctx_cho_jitter", getattr(C, "CTX_CHO_JITTER", 0.0)),
                ctx_ins_jitter=meta.get("ctx_ins_jitter", getattr(C, "CTX_INS_JITTER", 0.0)),
                ctx_resample=meta.get("ctx_resample", getattr(C, "CTX_RESAMPLE", True)),
                ctx_recompute_target=meta.get("ctx_recompute_target", True),
                model_kwargs=dict(self.model_kwargs),
                n_synth=hist["n_synth"], y0_lo=self.y0_lo, y0_hi=self.y0_hi,
                n_anchors=meta["n_anchors"], target_y_start=0.0, target_y_end=-amp,
            ),
        )
        return self.meta_
