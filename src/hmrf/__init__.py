"""Scalable-HMRF: scalable inference for hidden Markov random fields on spatial omics.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

The package exposes four modules:

* ``hmrf.paths``    -- repository layout and external-data resolution;
* ``hmrf.config``   -- YAML-backed configuration loader (no constants in code);
* ``hmrf.solvers``  -- mean-field VI, blocked Gibbs, emission models, ARI;
* ``hmrf.graphcut`` -- alpha-expansion graph cuts for Potts models;
* ``hmrf.simulate`` -- synthetic Potts/Gaussian/NB generators.

Library module names intentionally keep descriptive identifiers (no numeric
prefix) because Python import statements cannot start with a digit; every
entry-point script follows the ``NN_<verb>_<object>.py`` convention instead.
"""

__version__ = "1.0.0"
