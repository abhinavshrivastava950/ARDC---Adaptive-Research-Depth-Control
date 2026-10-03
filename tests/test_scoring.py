from cglc import scoring as S


def test_psi_and_utility_forms():
    f = S.ActionFeatures(delta=0.5, V=1.0, R=0.0, G=1.0, L=0.0, d=0.1, Pi=0.07)
    psi = S.structural_adjustment(f, 0.18, 0.14, 0.14, 0.10)
    assert abs(psi - (0.18 + 0.14)) < 1e-9
    assert abs(S.utility(f, 0.18, 0.14, 0.14, 0.10) - (0.5 + psi - 0.07)) < 1e-9


def test_value_per_budget_floor_and_eps():
    assert S.value_per_budget(-0.5, 0.1) == 0.0
    assert S.value_per_budget(0.5, 0.0, eps=1e-5) > 0


def test_delta_labels_fixed_mapping():
    m = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}
    assert S.delta_from_label("low", m) == 0.0
    assert S.delta_from_label("HIGH", m) == 1.0
