from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


def alpha_from_half_life(half_life: float) -> float:
    """
    Convert half-life (in steps) to EMA alpha.
    alpha = 1 - 2^(-1/half_life)
    """
    if half_life <= 0:
        raise ValueError("half_life must be > 0")
    return 1.0 - math.pow(2.0, -1.0 / half_life)


@dataclass
class EMAScore:
    alpha: float
    value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.alpha * float(x) + (1.0 - self.alpha) * self.value
        return float(self.value)
