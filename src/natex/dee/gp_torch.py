"""GPyTorch exact heteroskedastic-noise RBF GP -- the natex[gp] scale backend.

The numpy/scipy ``HeteroskedasticGP`` in ``dee/gp.py`` is the default
everywhere (DEE surfaces live on tens of centers; design spec section 10).
This backend exists only for scale (hundreds+ of centers, GPU): it exposes the
same surface -- construct with fixed hyperparameters, or :meth:`fit`, then
``posterior`` / ``log_marginal_likelihood`` -- and returns the same
``GPPosterior`` type, so the two are drop-in interchangeable.

Fitting uses ``botorch.fit.fit_gpytorch_mll`` -- the maintained API, NOT the
removed ``fit_gpytorch_model`` that broke the paper's repo on modern stacks
(repo risk 9; the ``gp`` extra therefore includes botorch). Model:
``FixedNoiseGaussianLikelihood`` (per-point noise variances known, no learned
extra noise) + ``ScaleKernel(RBFKernel())`` + ``ConstantMean``, matching the
numpy backend's kernel exactly.

Conventions (house rules, mirroring ``dee/gp.py``):

- torch is seeded from the caller's ``numpy.random.Generator``
  (``torch.manual_seed(int(rng.integers(2**63)))``); identical seed =>
  identical output. All tensors are float64 for parity with numpy.
- NaN never 0.0: fewer than 2 finite training rows => NaN mean/cov/MLL.
- ``posterior`` returns the LATENT posterior (noise not added), as numpy does.
- Exact computations only: fast/approximate prediction paths are disabled so
  both backends agree analytically at shared hyperparameters.

This module imports torch/gpytorch at the top: import it only when the
``natex[gp]`` extra is installed (tests importorskip; the core package never
imports it).
"""

from __future__ import annotations

import math
from typing import Any

import gpytorch
import numpy as np
import torch

from natex.dee.gp import GPPosterior, _as_2d, _require_rng

_LOG_2PI = math.log(2.0 * math.pi)


def _exact_settings():
    """Disable every approximate/cached prediction path (analytic parity)."""
    return gpytorch.settings.fast_computations(
        covar_root_decomposition=False, log_prob=False, solves=False
    )


class _ExactGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class TorchHeteroskedasticGP:
    """Exact RBF GP with constant mean and KNOWN per-point noise (gpytorch).

    Same surface and semantics as ``dee.gp.HeteroskedasticGP``; construct
    directly with fixed hyperparameters or via :meth:`fit`.
    """

    def __init__(
        self,
        lengthscale: float,
        outputscale: float,
        mean_const: float,
        X: np.ndarray,
        y: np.ndarray,
        noise_var: np.ndarray,
        fit_report: dict[str, Any] | None = None,
    ) -> None:
        self.lengthscale = float(lengthscale)
        self.outputscale = float(outputscale)
        self.mean_const = float(mean_const)
        self.X = _as_2d(X) if np.asarray(X).size else np.empty((0, 1))
        self.y = np.asarray(y, dtype=float).ravel()
        self.noise_var = np.broadcast_to(
            np.asarray(noise_var, dtype=float), self.y.shape
        ).astype(float)
        self.fit_report = fit_report
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f"X has {self.X.shape[0]} rows but y has {self.y.shape[0]}")
        if np.any(self.noise_var[np.isfinite(self.noise_var)] < 0.0):
            raise ValueError("noise_var must be nonnegative")
        self._model: _ExactGP | None = None
        if self._degenerate():
            return
        tx = torch.as_tensor(self.X, dtype=torch.float64)
        ty = torch.as_tensor(self.y, dtype=torch.float64)
        tn = torch.as_tensor(self.noise_var, dtype=torch.float64)
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=tn, learn_additional_noise=False
        )
        model = _ExactGP(tx, ty, likelihood).double()
        model.covar_module.base_kernel.lengthscale = self.lengthscale
        model.covar_module.outputscale = self.outputscale
        model.mean_module.constant = self.mean_const
        self._model = model

    def _degenerate(self) -> bool:
        return (
            self.X.shape[0] < 2
            or not np.isfinite(self.lengthscale)
            or not np.isfinite(self.outputscale)
            or not np.isfinite(self.mean_const)
            or not (np.all(np.isfinite(self.y)) and np.all(np.isfinite(self.noise_var)))
            or not np.all(np.isfinite(self.X))
        )

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        noise_var: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> "TorchHeteroskedasticGP":
        """Maximize the exact MLL via ``fit_gpytorch_mll``; deterministic given
        ``rng``. Non-finite rows are dropped (``fit_report['n_dropped']``);
        fewer than 2 finite rows => degenerate (NaN posterior/MLL, never 0.0).
        """
        rng = _require_rng(rng)
        try:
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import SingleTaskGP
        except ImportError:
            raise ImportError(
                "TorchHeteroskedasticGP.fit needs botorch (fit_gpytorch_mll); "
                'install the optional extra: pip install "natex-discovery[gp]"'
            ) from None
        torch.manual_seed(int(rng.integers(2**63)))
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        noise_var = np.broadcast_to(np.asarray(noise_var, dtype=float), y.shape)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
        finite = np.isfinite(y) & np.isfinite(noise_var) & np.all(np.isfinite(X), axis=1)
        Xf, yf, nvf = X[finite], y[finite], noise_var[finite].astype(float)
        report: dict[str, Any] = {
            "n_dropped": int((~finite).sum()),
            "n_used": int(finite.sum()),
            "best_mll": float("nan"),
            "backend": "gpytorch/fit_gpytorch_mll",
        }
        if Xf.shape[0] < 2:
            return cls(
                lengthscale=float("nan"),
                outputscale=float("nan"),
                mean_const=float("nan"),
                X=Xf,
                y=yf,
                noise_var=nvf,
                fit_report=report,
            )
        # fit_gpytorch_mll needs the botorch Model API; SingleTaskGP with
        # train_Yvar and no transforms IS FixedNoiseGaussianLikelihood + our
        # explicit RBF kernel. The fitted scalars are re-installed into the
        # plain gpytorch model below.
        tx = torch.as_tensor(Xf, dtype=torch.float64)
        ty = torch.as_tensor(yf, dtype=torch.float64)
        tn = torch.as_tensor(nvf, dtype=torch.float64)
        model = SingleTaskGP(
            train_X=tx,
            train_Y=ty.unsqueeze(-1),
            train_Yvar=tn.unsqueeze(-1),
            mean_module=gpytorch.means.ConstantMean(),
            covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
            outcome_transform=None,
            input_transform=None,
        ).double()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        with _exact_settings():
            fit_gpytorch_mll(mll)
        gp = cls(
            lengthscale=float(model.covar_module.base_kernel.lengthscale.item()),
            outputscale=float(model.covar_module.outputscale.item()),
            mean_const=float(model.mean_module.constant.item()),
            X=Xf,
            y=yf,
            noise_var=nvf,
            fit_report=report,
        )
        report["best_mll"] = gp.log_marginal_likelihood()
        return gp

    def log_marginal_likelihood(self) -> float:
        """Exact MLL at the stored hyperparameters; NaN when degenerate.

        gpytorch's ``ExactMarginalLogLikelihood`` divides by n; undo that so
        the value matches ``dee.gp.HeteroskedasticGP`` (total, with constants).
        """
        if self._model is None:
            return float("nan")
        model = self._model
        model.train()
        model.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        with torch.no_grad(), _exact_settings():
            out = model(model.train_inputs[0])
            val = mll(out, model.train_targets)
        return float(val.item()) * self.y.shape[0]

    def posterior(self, Xq: np.ndarray) -> GPPosterior:
        """LATENT posterior at ``Xq``; NaN mean/cov (never 0.0) when degenerate."""
        Xq = _as_2d(Xq)
        m = Xq.shape[0]
        if self._model is None:
            return GPPosterior(mean=np.full(m, np.nan), cov=np.full((m, m), np.nan))
        model = self._model
        model.eval()
        model.likelihood.eval()
        tq = torch.as_tensor(Xq, dtype=torch.float64)
        with (
            torch.no_grad(),
            gpytorch.settings.fast_pred_var(False),
            _exact_settings(),
        ):
            mvn = model(tq)  # latent f posterior: noise NOT added
            mean = mvn.mean.numpy().astype(float)
            cov = mvn.covariance_matrix.numpy().astype(float)
        cov = 0.5 * (cov + cov.T)  # symmetrize against roundoff
        return GPPosterior(mean=mean, cov=cov)
