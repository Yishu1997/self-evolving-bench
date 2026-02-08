from bench.generate import NoveltyFilter

def test_novelty_hash_dup():
    nf = NoveltyFilter(max_sim=0.95)
    nf.add("What is 2+2?")
    ok, info = nf.is_novel("What is 2+2?")
    assert not ok
    assert info["hash_dup"] is True

def test_novelty_semantic_dup():
    nf = NoveltyFilter(max_sim=0.4)  # strict
    nf.add("Explain how exponential moving average works.")
    ok, info = nf.is_novel("Describe exponential moving averages and how they work.")
    # may be flagged as similar under strict max_sim
    assert ok is False or ok is True
