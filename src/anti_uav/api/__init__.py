"""Operator console: a no-build static UI.

Deliberately plain HTML + vanilla JS. The system block diagram specifies a fully
offline edge deployment, so a UI with a build step, an npm registry dependency or
CDN-loaded assets is not deployable where it has to run. Chart.js is vendored into
``static/`` for that reason.

One HTML file, one stylesheet, one script. It reads the API and renders:

* environment status and what is actually converted
* the run x combo matrix, with the caveats that make a number interpretable
* per-run training curves, mAP50 and mAP50-95
* dataset statistics, including the bird-negative coverage column
* an inference playground over the val split
* the rule layer, with a "why was this not alerted" evaluator
* the 100-camera coverage map
"""

from __future__ import annotations

from .app import app

__all__ = ["app"]
