from server.blockarena_environment import BlockArenaEnvironment
from models import BlockArenaAction, ActionType

def test_reset():
    env = BlockArenaEnvironment("easy")
    obs = env.reset()
    assert obs.clauses_total == 4
    assert obs.clauses_agreed == 0
    assert obs.metadata["risk_level"] == "low"
    print("test_reset passed")

def test_probe_reward():
    env = BlockArenaEnvironment("easy")
    env.reset()
    r = env.step(BlockArenaAction(action_type=ActionType.PROBE, party="vendor", question="what?"))
    assert 0.10 <= r.reward <= 0.12, f"Expected reward ~0.10, got {r.reward}"
    assert r.metadata["next_best_action"] in {"ACCEPT", "PROPOSE", "PROBE"}
    assert r.metadata["action_mix"]["PROBE"] == 1
    print("test_probe_reward passed")

def test_propose_closes_clause():
    env = BlockArenaEnvironment("easy")
    env.reset()
    r = env.step(BlockArenaAction(action_type=ActionType.PROPOSE, clause_id="pricing", new_text="billed monthly"))
    assert r.clauses_agreed == 1
    print("test_propose_closes_clause passed")

def test_probe_budget_hard():
    env = BlockArenaEnvironment("hard")
    env.reset()
    for _ in range(3):
        env.step(BlockArenaAction(action_type=ActionType.PROBE, party="vendor", question="?"))
    r = env.step(BlockArenaAction(action_type=ActionType.PROBE, party="vendor", question="?"))
    assert "exhausted" in r.probe_result.lower()
    print("test_probe_budget passed")

def test_deterministic():
    env1 = BlockArenaEnvironment("easy")
    env2 = BlockArenaEnvironment("easy")
    obs1 = env1.reset()
    obs2 = env2.reset()
    assert obs1.clause_id == obs2.clause_id
    print("test_deterministic passed")


def test_summarize_action_does_not_consume_round():
    env = BlockArenaEnvironment("easy")
    env.reset()
    r = env.step(BlockArenaAction(action_type=ActionType.SUMMARIZE, clause_id="pricing"))
    assert r.round_number == 0
    assert "Progress:" in (r.probe_result or "")
    print("test_summarize_action_does_not_consume_round passed")

if __name__ == "__main__":
    test_reset()
    test_probe_reward()
    test_propose_closes_clause()
    test_probe_budget_hard()
    test_deterministic()
    test_summarize_action_does_not_consume_round()
    print("All tests passed")