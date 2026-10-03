from cglc import scoring as S
from cglc.config import DEFAULT_CONFIG
from cglc.controller import rank_actions, decide
from cglc.gates import evaluate_finalization


def _feats():
    return {
        "CONTINUE": S.ActionFeatures(delta=0.5, G=1.0, d=0.05),
        "VERIFY": S.ActionFeatures(delta=0.0, d=0.03),
        "REDIRECT": S.ActionFeatures(delta=0.2, R=0.5, d=0.08),
    }


def test_ranking_never_overrides_gates():
    # Even with high r_t, gates decide finalization first.
    ranked = rank_actions(_feats(), DEFAULT_CONFIG, rho=0.1,
                          remaining={"tool_calls": 9, "tokens": 1e4, "wall_clock": 1e3},
                          allowed_classes=["CONTINUE", "VERIFY", "REDIRECT"],
                          blockers=[])
    gates = evaluate_finalization(True, True, True, True, False)
    d, _ = decide(gates, False, False, ranked)
    assert d == "ALLOW_FINALIZE"


def test_fallback_verify_on_contested():
    ranked = rank_actions(
        {
            "CONTINUE": S.ActionFeatures(delta=0.0, L=1.0, d=0.05),
            "VERIFY": S.ActionFeatures(delta=0.0, V=0.5, d=0.03),
        },
        DEFAULT_CONFIG, rho=0.9,
        remaining={"tool_calls": 9, "tokens": 1e4, "wall_clock": 1e3},
        allowed_classes=["CONTINUE", "VERIFY"], blockers=[])
    gates = evaluate_finalization(True, True, False, True, False)
    d, _ = decide(gates, False, False, ranked, weak_or_contested=True)
    assert d == "VERIFY"


def test_low_value_never_forces_finalize():
    ranked = rank_actions(
        {"CONTINUE": S.ActionFeatures(delta=0.0, L=1.0, d=0.05, Pi=5.0)},
        DEFAULT_CONFIG, rho=1.0,
        remaining={"tool_calls": 9, "tokens": 1e4, "wall_clock": 1e3},
        allowed_classes=["CONTINUE"], blockers=[])
    gates = evaluate_finalization(True, True, False, True, False)
    d, _ = decide(gates, False, False, ranked)
    assert d != "ALLOW_FINALIZE"
