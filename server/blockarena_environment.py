import json
import re
from pathlib import Path
from uuid import uuid4
from typing import Any, Dict

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from openenv.core.rubrics.base import Rubric

try:
    from ..models import BlockArenaAction, BlockArenaObservation
    from .opponents import VendorAgent, LegalReviewer
except ImportError:
    from models import BlockArenaAction, BlockArenaObservation
    from server.opponents import VendorAgent, LegalReviewer

DEALS_DIR = Path(__file__).parent / "deals"


def load_deal(tier: str) -> Dict[str, Any]:
    path = DEALS_DIR / f"{tier}.json"
    if not path.exists():
        raise FileNotFoundError(f"Deal file not found for tier '{tier}': {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clamp(value: float) -> float:
    return round(min(max(value, 0.01), 0.99), 4)


def safe_score(value: float) -> float:
    # Clamp to (0.001, 0.999) before mapping to guarantee strict (0,1) output
    value = min(max(value, 0.001), 0.999)
    result = round(0.01 + 0.98 * value, 4)
    # Final hard clamp — belt and suspenders
    return min(max(result, 0.01), 0.99)


class BlockArenaRubric(Rubric):
    def __init__(self, env: "BlockArenaEnvironment"):
        super().__init__()
        self._env = env

    def forward(self, action: Any, observation: Any) -> float:
        rewards = self._env._episode_rewards
        if not rewards:
            return 0.01
        raw = sum(rewards)
        max_possible = max(len(self._env._clauses) * 0.40 + 0.40, 0.01)
        normalized = raw / max_possible
        return safe_score(normalized)

    def reset(self) -> None:
        pass


class BlockArenaEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, tier: str = "easy"):
        self._tier = tier
        self._deal = load_deal(tier)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._init_opponents()
        self._reset_episode()
        self.rubric = BlockArenaRubric(self)

    def _init_opponents(self) -> None:
        vendor_hidden = self._deal["vendor_hidden"]
        legal_hidden = self._deal["legal_hidden"]
        self._vendor = VendorAgent(
            hidden_priority=vendor_hidden["priority"],
            hidden_value=vendor_hidden["value"],
            walkout_threshold=vendor_hidden.get("walkout_threshold", 3),
        )
        self._legal = LegalReviewer(
            hidden_redline=legal_hidden["redline"],
            hidden_value=legal_hidden["value"],
            flagged_patterns=legal_hidden["flagged_patterns"],
        )

    def _reset_episode(self) -> None:
        self._clauses = list(self._deal["clauses"])
        self._clause_index = 0
        self._agreed = {}
        self._round_budget = int(self._deal["round_budget"])
        self._probe_budget = int(self._deal.get("probe_budget", 999))
        self._probes_used = 0
        self._rounds_used = 0
        self._episode_rewards = []
        self._action_counts = {
            "ACCEPT": 0,
            "REJECT": 0,
            "PROPOSE": 0,
            "PROBE": 0,
            "ESCALATE": 0,
            "SUMMARIZE": 0,
        }
        self._last_vendor_stance = "open"
        self._last_legal_stance = "approved"
        self._vendor.reset()
        self._legal.reset()

    def reset(self) -> BlockArenaObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_episode()
        if self.rubric is not None:
            self.rubric.reset()
        clause = self._clauses[0]
        return BlockArenaObservation(
            clause_id=clause["id"],
            clause_text=clause["text"],
            vendor_response="Vendor is ready to negotiate.",
            legal_response="Legal is ready to review.",
            probe_result=None,
            round_number=0,
            rounds_remaining=self._round_budget,
            clauses_agreed=0,
            clauses_total=len(self._clauses),
            tier=self._tier,
            done=False,
            reward=0.01,
            metadata={
                "vendor_stance": "open",
                "legal_stance": "approved",
                "agreed_clauses": [],
                "episode_score": 0.01,
                "probes_remaining": self._probe_budget,
                "risk_level": "low",
                "next_best_action": "PROBE",
                "action_mix": self._action_counts,
            },
        )

    def step(self, action: BlockArenaAction) -> BlockArenaObservation:
        self._state.step_count += 1

        clause = self._clauses[self._clause_index]
        clause_id = clause["id"]
        new_text = action.new_text or clause["text"]
        action_type = action.action_type.value

        if action_type in self._action_counts:
            self._action_counts[action_type] += 1

        if action_type == "SUMMARIZE":
            score = self._compute_episode_score()
            summary = self._build_progress_summary(score)
            return self._build_observation(
                clause=clause,
                vendor_resp="Negotiation summary generated.",
                legal_resp="No legal updates during summary.",
                probe_result=summary,
                vendor_stance=self._last_vendor_stance,
                legal_stance=self._last_legal_stance,
                reward=0.03,
                done=False,
                score=score,
            )

        self._rounds_used += 1

        vendor_resp, vendor_stance = self._vendor.respond(action_type, clause_id, new_text)
        legal_resp, legal_stance = self._legal.review(action_type, clause_id, new_text)
        self._last_vendor_stance = vendor_stance
        self._last_legal_stance = legal_stance

        probe_result = None
        if action_type == "PROBE":
            if self._probes_used >= self._probe_budget:
                probe_result = "Probe budget exhausted — no more information available."
                vendor_resp = "No further information."
                legal_resp = "No further information."
            else:
                self._probes_used += 1
                party = (action.party or "vendor").lower()
                probe_result = vendor_resp if party == "vendor" else legal_resp

        reward = self._calculate_reward(
            action_type=action_type,
            vendor_stance=vendor_stance,
            legal_stance=legal_stance,
            probe_result=probe_result,
        )

        if vendor_stance == "open" and legal_stance == "approved":
            if action_type in ("ACCEPT", "PROPOSE"):
                self._agreed[clause_id] = new_text
                self._clause_index = min(self._clause_index + 1, len(self._clauses) - 1)

        self._episode_rewards.append(clamp(reward))

        all_agreed = len(self._agreed) == len(self._clauses)
        walkout = vendor_stance == "walkout"
        out_of_rounds = self._rounds_used >= self._round_budget
        done = all_agreed or walkout or out_of_rounds

        if done:
            bonus = self._calculate_final_bonus()
            reward = clamp(reward + bonus)
            self._episode_rewards[-1] = reward

        reward = clamp(reward)

        score = self._compute_episode_score()

        if not done and self._clause_index < len(self._clauses):
            next_clause = self._clauses[self._clause_index]
        else:
            next_clause = clause

        return self._build_observation(
            clause=next_clause,
            vendor_resp=vendor_resp,
            legal_resp=legal_resp,
            probe_result=probe_result,
            vendor_stance=vendor_stance,
            legal_stance=legal_stance,
            reward=reward,
            done=done,
            score=score,
        )

    def _build_observation(
        self,
        clause: Dict[str, Any],
        vendor_resp: str,
        legal_resp: str,
        probe_result: str | None,
        vendor_stance: str,
        legal_stance: str,
        reward: float,
        done: bool,
        score: float,
    ) -> BlockArenaObservation:
        return BlockArenaObservation(
            clause_id=clause["id"],
            clause_text=clause["text"],
            vendor_response=vendor_resp,
            legal_response=legal_resp,
            probe_result=probe_result,
            round_number=self._rounds_used,
            rounds_remaining=max(self._round_budget - self._rounds_used, 0),
            clauses_agreed=len(self._agreed),
            clauses_total=len(self._clauses),
            tier=self._tier,
            done=done,
            reward=reward,
            metadata={
                "vendor_stance": vendor_stance,
                "legal_stance": legal_stance,
                "agreed_clauses": list(self._agreed.keys()),
                "episode_score": score,
                "probes_remaining": max(self._probe_budget - self._probes_used, 0),
                "risk_level": self._risk_level(vendor_stance, legal_stance),
                "next_best_action": self._next_best_action(vendor_stance, legal_stance),
                "action_mix": self._action_counts,
            },
        )

    def _compute_episode_score(self) -> float:
        raw_total = sum(self._episode_rewards)
        max_possible = max(len(self._clauses) * 0.40 + 0.40, 0.01)
        normalized = raw_total / max_possible
        return safe_score(normalized)

    def _risk_level(self, vendor_stance: str, legal_stance: str) -> str:
        if vendor_stance == "walkout" or legal_stance == "flagged":
            return "high"
        if vendor_stance == "firm" or self._rounds_used >= int(self._round_budget * 0.7):
            return "medium"
        return "low"

    def _next_best_action(self, vendor_stance: str, legal_stance: str) -> str:
        if legal_stance == "flagged":
            return "PROPOSE"
        if vendor_stance == "firm" and self._probes_used < self._probe_budget:
            return "PROBE"
        if vendor_stance == "open" and legal_stance == "approved":
            return "ACCEPT"
        return "PROPOSE"

    def _build_progress_summary(self, score: float) -> str:
        return (
            f"Progress: {len(self._agreed)}/{len(self._clauses)} clauses agreed; "
            f"rounds used {self._rounds_used}/{self._round_budget}; "
            f"probes used {self._probes_used}/{self._probe_budget}; "
            f"current score {score:.4f}."
        )

    def _calculate_reward(
        self,
        action_type: str,
        vendor_stance: str,
        legal_stance: str,
        probe_result: str | None,
    ) -> float:
        reward = 0.01
        if vendor_stance == "open" and legal_stance == "approved":
            if action_type in ("ACCEPT", "PROPOSE"):
                reward += 0.40
        if action_type == "PROBE" and probe_result:
            reward += 0.10
        if legal_stance == "flagged":
            reward -= 0.20
        if vendor_stance == "walkout":
            reward -= 0.30
        return clamp(reward)

    def _calculate_final_bonus(self) -> float:
        bonus = 0.0
        agreed_text = " ".join(self._agreed.values()).lower()
        vendor_hidden = self._deal["vendor_hidden"]
        if vendor_hidden["value"].lower() in agreed_text:
            bonus += 0.20
        legal_hidden = self._deal["legal_hidden"]
        has_redline = any(
            re.search(pattern, agreed_text, re.IGNORECASE)
            for pattern in legal_hidden["flagged_patterns"]
        )
        if not has_redline:
            bonus += 0.15
        if self._rounds_used < self._round_budget:
            bonus += 0.05
        return clamp(bonus)

    @property
    def state(self) -> State:
        return self._state


# Backward-compatible aliases.
ContractArenaRubric = BlockArenaRubric
ContractarenaEnvironment = BlockArenaEnvironment