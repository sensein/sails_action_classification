"""bootstrap_ci_d shared exclusively between handflapping.py and spinning.py (their cohen_d dependency is verified identical to handflapping/jumping/spinning's shared cohen_d_v3)."""

import numpy as np

from sailsprep.analysis.common.effect_size import cohen_d_v3 as cohen_d


def bootstrap_ci_d(a, b, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    boot = [cohen_d(rng.choice(a, len(a), replace=True),
                    rng.choice(b, len(b), replace=True))
            for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
