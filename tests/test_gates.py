from cglc.gates import evaluate_finalization, halt_all_match, coverage_ok


def test_conjunctive_gate():
    g = evaluate_finalization(True, True, True, True, False)
    assert g.allow is True
    for kw in [
        dict(has_final_candidate=False, process_complete=True,
             evidence_sufficient=True, answer_conforms=True, blocker_present=False),
        dict(has_final_candidate=True, process_complete=False,
             evidence_sufficient=True, answer_conforms=True, blocker_present=False),
        dict(has_final_candidate=True, process_complete=True,
             evidence_sufficient=False, answer_conforms=True, blocker_present=False),
        dict(has_final_candidate=True, process_complete=True,
             evidence_sufficient=True, answer_conforms=False, blocker_present=False),
        dict(has_final_candidate=True, process_complete=True,
             evidence_sufficient=True, answer_conforms=True, blocker_present=True),
    ]:
        assert evaluate_finalization(**kw).allow is False


def test_halt_all_match_defaults():
    assert halt_all_match([True, True], tau=0.0) is True
    assert halt_all_match([True, False]) is False
    assert halt_all_match([]) is False


def test_coverage_theta():
    assert coverage_ok(85, 100, 0.85) is True
    assert coverage_ok(84, 100, 0.85) is False
