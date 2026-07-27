"""
vis/curve_fitter.py — Multi-stage robust curve fitting utility.

Pipeline stages
---------------
1. Normalize variables (x_scale = max|x|, y_scale = max|y|) — used only
   to compute a stable initial guess; the nonlinear fit runs on original data.
2. Log-linear OLS to bootstrap an initial guess for power-exponential models:
       ln(y) = ln(a) + b·ln(x) − k·x    (k = ln(c)/d)
   Falls back to an analytical peak-based estimate when OLS is unavailable.
3. Bounded nonlinear fit via scipy curve_fit (trf when bounds supplied);
   differential_evolution fallback when curve_fit fails and bounds are finite.
4. Identifiability check: flag high parameter correlations (|r|>0.95) and
   parameters whose standard error exceeds their value.
5. Residual diagnostics: R², RMSE, AIC/BIC, sign-change structure check.

Intended model form
-------------------
    y = a · x^b · c^(−x/d)          [4-param power-exponential]
    y = a · x^b · exp(−x/d)         [3-param variant, c fixed at e]

The rescaling relations from normalised → physical space assume this layout:
    a_phys = a_norm · y_scale / x_scale^b
    b_phys = b_norm
    c_phys = c_norm
    d_phys = d_norm · x_scale

Usage
-----
    from curve_fitter import fit_curve
    result = fit_curve(my_func, xdata, ydata,
                       bounds=([0, 0, 1.001, 0], [1e8, 10, 100, 1e5]))
    if result.converged:
        popt, r2 = result.popt, result.r2
"""

import inspect
import numpy as np
from dataclasses import dataclass, field
from scipy import optimize
from typing import Callable, List, Optional, Tuple


@dataclass
class FitResult:
    """All outputs from a single fit_curve call."""
    popt: Optional[np.ndarray] = None       # fitted parameters (physical units)
    pcov: Optional[np.ndarray] = None       # parameter covariance matrix
    r2: Optional[float] = None              # coefficient of determination
    rmse: Optional[float] = None            # root-mean-square error (physical units)
    aic: Optional[float] = None             # Akaike information criterion
    bic: Optional[float] = None             # Bayesian information criterion
    residuals: Optional[np.ndarray] = None  # ydata − y_fit
    identifiable: bool = True               # False if params are correlated / unconstrained
    converged: bool = False
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 2 — initial-guess helpers
# ---------------------------------------------------------------------------

def _log_linear_p0(
    xn: np.ndarray,
    yn: np.ndarray,
    n_params: int,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Log-linear OLS on normalised data for power-exponential models.

    Returns (p0_norm, n_dropped).  p0_norm is in normalised space;
    caller must rescale to physical space before passing to curve_fit.
    """
    mask = yn > 0
    n_dropped = int(np.sum(~mask))
    xp, yp = xn[mask], yn[mask]

    if len(xp) < 3:
        return None, n_dropped

    logy = np.log(yp)
    A = np.column_stack([np.ones_like(xp), np.log(xp), xp])
    coeffs, _, rank, _ = np.linalg.lstsq(A, logy, rcond=None)

    if rank < 3:
        return None, n_dropped

    ln_a, b, neg_k = coeffs
    k = -neg_k
    if k <= 0:          # decay coefficient must be positive
        return None, n_dropped

    a = np.exp(ln_a)
    d = 1.0 / k         # c fixed at e → ln(c)=1, so k = 1/d

    if n_params == 3:
        return np.array([a, b, d]), n_dropped
    elif n_params == 4:
        return np.array([a, b, np.e, d]), n_dropped
    else:
        base = [a, b, np.e, d]
        pad  = [1.0] * max(0, n_params - 4)
        return np.array((base + pad)[:n_params]), n_dropped


def _analytical_p0(xdata: np.ndarray, ydata: np.ndarray, n_params: int) -> np.ndarray:
    """
    Fallback p0 derived from peak position and height.

    Assumes n=1, sets τ = x_peak, solves for amplitude a.
    """
    i_peak   = int(np.argmax(ydata))
    x_peak   = float(xdata[i_peak])
    y_peak   = float(ydata[i_peak])
    n        = 1.0
    tau      = max(x_peak, 1e-6)
    a        = y_peak * np.e / tau   # from y(x*) = a·τ·e^{-1}

    if n_params == 3:
        return np.array([a, n, tau])
    elif n_params == 4:
        return np.array([a, n, np.e, tau])
    else:
        base = [a, n, np.e, tau]
        pad  = [1.0] * max(0, n_params - 4)
        return np.array((base + pad)[:n_params])


def _rescale_p0(p0_norm: np.ndarray, x_scale: float, y_scale: float) -> np.ndarray:
    """
    Convert normalised p0 to physical-unit p0 for the power-exponential form.

        a_phys = a_norm · y_scale / x_scale^b
        b, c   unchanged
        d_phys = d_norm · x_scale
    """
    p0 = p0_norm.copy().astype(float)
    b     = float(p0_norm[1])
    p0[0] = p0_norm[0] * y_scale / (x_scale ** b)
    if len(p0) >= 4:
        p0[3] = p0_norm[3] * x_scale
    elif len(p0) == 3:
        p0[2] = p0_norm[2] * x_scale
    return p0


# ---------------------------------------------------------------------------
# Stage 4 — identifiability
# ---------------------------------------------------------------------------

def _check_identifiability(
    pcov: Optional[np.ndarray],
    popt: np.ndarray,
    warnings: List[str],
) -> bool:
    if pcov is None or np.any(np.isinf(pcov)):
        warnings.append('Covariance matrix is infinite — parameters may be unidentifiable.')
        return False

    diag = np.diag(pcov)
    if np.any(diag <= 0):
        warnings.append('Covariance matrix has non-positive diagonal — fit may be degenerate.')
        return False

    std_errs     = np.sqrt(diag)
    identifiable = True

    for i, (se, pv) in enumerate(zip(std_errs, popt)):
        if abs(pv) > 0 and se > abs(pv):
            warnings.append(
                f'Param {i}: std error {se:.3g} > |value| {abs(pv):.3g} — effectively unconstrained.'
            )
            identifiable = False

    outer = np.outer(std_errs, std_errs)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.where(outer > 0, pcov / outer, 0.0)

    for i in range(len(popt)):
        for j in range(i + 1, len(popt)):
            if abs(corr[i, j]) > 0.95:
                warnings.append(
                    f'Params {i}/{j} correlation = {corr[i, j]:.3f} — consider a simpler model.'
                )
                identifiable = False

    return identifiable


# ---------------------------------------------------------------------------
# Stage 5 — residual diagnostics
# ---------------------------------------------------------------------------

def _compute_diagnostics(
    func: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    popt: np.ndarray,
) -> Tuple[float, float, float, float, np.ndarray]:
    """Return (r2, rmse, aic, bic, residuals) in original units."""
    yfit      = func(xdata, *popt)
    residuals = ydata - yfit
    ss_res    = float(np.sum(residuals ** 2))
    ss_tot    = float(np.sum((ydata - np.mean(ydata)) ** 2))
    n, k      = len(ydata), len(popt)

    r2   = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    rmse = np.sqrt(ss_res / n)

    if ss_res > 0 and n > k:
        log_lik = -n / 2.0 * np.log(ss_res / n)
        aic = 2.0 * k - 2.0 * log_lik
        bic = k * np.log(n) - 2.0 * log_lik
    else:
        aic = bic = float('nan')

    return r2, rmse, aic, bic, residuals


def _check_residual_structure(
    xdata: np.ndarray,
    residuals: np.ndarray,
    warnings: List[str],
) -> None:
    """Flag systematic trends: too few sign changes in residuals sorted by x."""
    if len(residuals) < 4:
        return
    r     = residuals[np.argsort(xdata)]
    signs = np.sign(r[r != 0])
    if len(signs) < 2:
        return
    n_changes = int(np.sum(signs[:-1] != signs[1:]))
    threshold = max(1, len(signs) // 4)
    if n_changes < threshold:
        warnings.append(
            f'Residuals have {n_changes} sign change(s) across {len(signs)} non-zero '
            'points — possible systematic model misfit.'
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_curve(
    func: Callable,
    xdata,
    ydata,
    p0=None,
    bounds=None,
    linearize: bool = True,
    sigma=None,
    global_fallback: bool = True,
    verbose: bool = True,
) -> FitResult:
    """
    Multi-stage robust curve fit.

    Parameters
    ----------
    func : callable
        Model function f(x, *params).
    xdata, ydata : array_like
        Observed data — 1-D, no NaNs.
    p0 : array_like, optional
        Initial parameter guess.  Auto-computed when None.
    bounds : (array_like, array_like), optional
        (lower_bounds, upper_bounds).  Enables the bounded trf solver and,
        when all bounds are finite, the differential_evolution fallback.
    linearize : bool
        Use log-linear OLS to compute p0 (power-exponential models).
        Falls back to analytical peak estimate on failure.
    sigma : array_like, optional
        Per-point standard deviations passed to curve_fit.
    global_fallback : bool
        Try differential_evolution when curve_fit fails (needs finite bounds).
    verbose : bool
        Print stage progress and diagnostics.

    Returns
    -------
    FitResult
        .converged is True only when a numerical solution was found.
        .warnings lists all identifiability and structural concerns.
    """
    xdata  = np.asarray(xdata, dtype=float).ravel()
    ydata  = np.asarray(ydata, dtype=float).ravel()
    result = FitResult()

    if len(xdata) < 2:
        result.warnings.append('Insufficient data points.')
        return result

    if bounds is not None:
        bounds = (np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float))

    # Infer parameter count ------------------------------------------------
    if p0 is not None:
        n_params = len(p0)
    elif bounds is not None:
        n_params = len(bounds[0])
    else:
        n_params = len(inspect.signature(func).parameters) - 1

    # Stage 1: normalization (used only for p0 computation) ----------------
    x_scale = float(np.max(np.abs(xdata))) or 1.0
    y_scale = float(np.max(np.abs(ydata))) or 1.0
    xn = xdata / x_scale
    yn = ydata / y_scale

    # Stage 2: initial guess -----------------------------------------------
    if p0 is None:
        p0_guess = None
        if linearize:
            p0_norm, n_dropped = _log_linear_p0(xn, yn, n_params)
            if n_dropped > 0 and verbose:
                print(f'  [curve_fitter] log-linear OLS: dropped {n_dropped} non-positive y point(s).')
            if p0_norm is not None:
                p0_guess = _rescale_p0(p0_norm, x_scale, y_scale)
                if verbose:
                    print(f'  [curve_fitter] log-linear p0: {np.round(p0_guess, 4)}')
            elif verbose:
                print('  [curve_fitter] log-linear OLS failed; using analytical peak estimate.')

        if p0_guess is None:
            p0_guess = _analytical_p0(xdata, ydata, n_params)
            if verbose:
                print(f'  [curve_fitter] analytical p0: {np.round(p0_guess, 4)}')

        p0 = p0_guess

    # Stage 3: bounded nonlinear fit ---------------------------------------
    cf_kwargs: dict = {'p0': p0, 'maxfev': 100_000}
    if bounds is not None:
        cf_kwargs['bounds'] = bounds
        cf_kwargs.pop('maxfev', None)
    if sigma is not None:
        cf_kwargs['sigma']           = sigma
        cf_kwargs['absolute_sigma']  = True

    try:
        popt, pcov = optimize.curve_fit(func, xdata, ydata, **cf_kwargs)
        result.converged = True
        if verbose:
            print('  [curve_fitter] curve_fit converged.')
    except (RuntimeError, ValueError) as exc:
        if verbose:
            print(f'  [curve_fitter] curve_fit failed: {exc}')

        # Differential-evolution fallback (finite bounds required) ----------
        bounds_finite = (
            global_fallback
            and bounds is not None
            and np.all(np.isfinite(bounds[0]))
            and np.all(np.isfinite(bounds[1]))
        )
        if bounds_finite:
            if verbose:
                print('  [curve_fitter] Trying differential_evolution…')
            try:
                de_bounds = list(zip(bounds[0], bounds[1]))
                de_res = optimize.differential_evolution(
                    lambda params: np.sum((func(xdata, *params) - ydata) ** 2),
                    bounds=de_bounds,
                    seed=42,
                    maxiter=2000,
                    tol=1e-9,
                    polish=True,
                )
                if de_res.success:
                    try:
                        popt, pcov = optimize.curve_fit(
                            func, xdata, ydata,
                            p0=de_res.x, bounds=bounds, maxfev=100_000,
                        )
                    except Exception:
                        popt = de_res.x
                        pcov = None
                    result.converged = True
                    if verbose:
                        print('  [curve_fitter] differential_evolution converged.')
                else:
                    result.warnings.append('differential_evolution did not converge.')
                    return result
            except Exception as exc2:
                result.warnings.append(f'differential_evolution error: {exc2}')
                return result
        else:
            result.warnings.append(f'curve_fit failed: {exc}')
            return result

    result.popt = popt
    result.pcov = pcov

    # Stage 4: identifiability check ----------------------------------------
    result.identifiable = _check_identifiability(pcov, popt, result.warnings)

    # Stage 5: residual diagnostics -----------------------------------------
    r2, rmse, aic, bic, residuals = _compute_diagnostics(func, xdata, ydata, popt)
    result.r2        = r2
    result.rmse      = rmse
    result.aic       = aic
    result.bic       = bic
    result.residuals = residuals
    _check_residual_structure(xdata, residuals, result.warnings)

    if verbose:
        print(f'  [curve_fitter] R²={r2:.4f}  RMSE={rmse:.4g}  AIC={aic:.4g}  BIC={bic:.4g}')
        for w in result.warnings:
            print(f'  [curve_fitter] WARNING: {w}')

    return result
