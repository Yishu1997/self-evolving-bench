import math
from bench.ema import alpha_from_half_life, EMAScore

def test_alpha():
    a = alpha_from_half_life(10)
    assert 0 < a < 1

def test_ema_update():
    ema = EMAScore(alpha=0.5)
    assert ema.update(1.0) == 1.0
    assert math.isclose(ema.update(0.0), 0.5)
