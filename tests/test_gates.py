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


def test_authorizer_certificate_and_rejection():
    from cglc.gates import FinalizationAuthorizer
    from cglc.contracts import TaskContract
    from cglc.ledger import EvidenceLedger
    auth = FinalizationAuthorizer()
    c = TaskContract.create("g", ["duty"], ["claim"])
    led = EvidenceLedger(["ev-0"])
    ok = evaluate_finalization(True, True, True, True, False)
    cert = auth.authorize(ok, c, led, checkpoint_id=3)
    assert cert.allowed is True
    assert cert.certificate["contract"]["contract_id"] == c.contract_id
    assert cert.certificate["checkpoint_id"] == 3
    bad = evaluate_finalization(True, True, False, True, False)
    rej = auth.authorize(bad, c, led, checkpoint_id=3)
    assert rej.allowed is False and rej.reasons != []
    assert rej.certificate is None


def test_contract_validate_and_json():
    from cglc.contracts import TaskContract
    bad = TaskContract.create("", [], [])
    assert "empty goal" in bad.validate()
    assert "no hard obligations" in " ".join(bad.validate())
    good = TaskContract.create("g", ["d"], ["e"])
    assert good.validate() == []
    d = good.to_dict()
    assert d["provenance"] and d["revision"] == 1 and d["goal"] == "g"
