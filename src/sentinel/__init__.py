"""Smoke Sentinel: early wildfire smoke detection on fixed camera networks.

Modules
-------
``data``        FIgLib access, caching and filename-derived labels.
``background``  Anchored long-baseline reference building and alignment.
``detector``    Classical OpenCV 5 residual scoring and annotation.
``features``    The twelve per-frame measurements fed to the fitted boundary.
``agent``       Verification policy and guardrails (measured, not enabled).

Submodules are imported lazily by callers; importing this package pulls in
nothing heavy, so ``import sentinel`` stays cheap in a Lambda cold start.
"""

__version__ = "0.1.0"
