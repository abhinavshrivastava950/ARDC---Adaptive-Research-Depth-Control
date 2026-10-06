from cglc import budget as B


def test_pressure_approaches_one_when_scarce():
    lim = {"tool_calls": 10.0, "tokens": 100.0}
    assert B.remaining_pressure({"tool_calls": 10.0, "tokens": 100.0}, lim) == 0.0
    assert abs(B.remaining_pressure({"tool_calls": 0.0, "tokens": 100.0}, lim) - 1.0) < 1e-9
    # min-clip semantics: scarcest dimension dominates
    assert abs(B.remaining_pressure({"tool_calls": 5.0, "tokens": 100.0}, lim) - 0.5) < 1e-9


def test_normalized_cost_equal_weights():
    lim = {"tool_calls": 10.0, "tokens": 100.0}
    om = {"tool_calls": 0.5, "tokens": 0.5}
    d = B.normalized_cost({"tool_calls": 1.0, "tokens": 10.0}, lim, om)
    assert abs(d - 0.1) < 1e-9


def test_action_pressure_form():
    assert abs(B.action_pressure(0.7, 0.5, 0.2) - 0.07) < 1e-9


def test_lease_budget_cap_scales_with_the_remaining_budget():
    from cglc.config import DEFAULT_CONFIG
    from cglc.leases import lease_budget_cap
    shares = DEFAULT_CONFIG.lease.budget_share
    full = lease_budget_cap("STANDARD", 3, {"tool_calls": 30.0, "tokens": 60000.0, "wall_clock": 600.0}, shares)
    half = lease_budget_cap("STANDARD", 3, {"tool_calls": 15.0, "tokens": 30000.0, "wall_clock": 300.0}, shares)
    assert full == {"tool_calls": 3.0, "tokens": 21000.0, "wall_clock": 210.0}
    assert half["tokens"] == full["tokens"] / 2 and half["tool_calls"] == 3.0
    assert lease_budget_cap("STANDARD", 3, {"tokens": -5.0}, shares)["tokens"] == 0.0  # overspent: cap floors at 0
