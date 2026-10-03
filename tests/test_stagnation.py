from cglc.stagnation import StagnationTracker, unique_passage_rate, jaccard, tokens


def test_upr_is_novel_over_current_not_history():
    assert unique_passage_rate(["a", "b", "c"], {"b"}) == 2 / 3
    assert unique_passage_rate([], {"a"}) == 0.0
    assert unique_passage_rate(["a"], set()) == 1.0


def test_jaccard_edges():
    assert jaccard(tokens(""), tokens("")) == 1.0
    assert jaccard(tokens("a b"), tokens("")) == 0.0
    assert abs(jaccard(tokens("a b"), tokens("b c")) - 1 / 3) < 1e-9


def test_stagnation_needs_p_consecutive_cgdp_defaults():
    tr = StagnationTracker(tau_J=0.6, tau_U=0.3, p=2, recent_window=5)
    _, _, s1 = tr.step("search architecture comparison", ["d1#0"])
    assert s1 is False  # fewer than p rounds
    # repeat same action, no novel chunks -> J=1.0, UPR=0.0: 1st qualifying round
    _, J2, s2 = tr.step("search architecture comparison", ["d1#0"])
    assert s2 is False  # only one consecutive qualifying round so far
    # third identical repeat -> p=2 consecutive qualifying rounds
    _, _, s3 = tr.step("search architecture comparison", ["d1#0"])
    assert s3 is True


def test_novelty_breaks_stagnation():
    tr = StagnationTracker(tau_J=0.6, tau_U=0.3, p=2)
    tr.step("search alpha", ["a"])
    _J, _U, s = tr.step("search alpha", ["b", "c"])
    # J high but UPR=1.0 > tau_U -> not stagnated
    assert s is False
