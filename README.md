---
title: BlockArena
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
---

# BlockArena

A strategic multi-party contract negotiation environment built for OpenEnv.

**Built with precision for the OpenEnv Hackathon 2026.**

## Team

**Codecatalysts -- OpenEnv Hackathon 2026**

- **Ayush** — Team Lead
- **Yatharth Gautam**
- **Vardaan Dua**

## What Is BlockArena?

BlockArena is a partially observable reinforcement learning environment where an agent negotiates contract clauses against two opposing stakeholders:

- **VendorAgent**: Protects business priorities and may walk out if pushed too hard
- **LegalReviewer**: Enforces compliance redlines and flags problematic terms

The objective is to close as many clauses as possible while maximizing episode-level score through strategic probing and negotiation.

## New Features

### 1. SUMMARIZE Action
- New action type that generates in-episode negotiation progress summary
- Does not consume negotiation rounds
- Helps agents checkpoint strategy before the next move
- Returns summary in `probe_result` field

### 2. Strategy Guidance Metadata
Every observation now includes tactical guidance:
- `risk_level`: low | medium | high context-aware risk signal
- `next_best_action`: AI-suggested next move (ACCEPT, PROBE, PROPOSE, etc.)
- `action_mix`: Running count of each action used in episode

## Action Space

| Action | Purpose |
|--------|---------|
| ACCEPT | Agree to current clause as-is |
| REJECT | Reject clause with optional reason |
| PROPOSE | Counter-propose new clause text |
| PROBE | Learn hidden requirements from vendor/legal |
| ESCALATE | Force escalation (triggers legal flag) |
| SUMMARIZE | Get negotiation progress summary |

## Difficulty Tiers

| Tier | Clauses | Probes | Rounds | Challenge |
|------|---------|--------|---------|-----------|
| Easy | 4 | Unlimited | 8 | Single hidden agenda |
| Medium | 8 | Unlimited | 15 | Two opposing agendas |
| Hard | 12 | **3 only** | 20 | Conflicting requirements |

## Quick Start

```bash
# Clone repository
git clone https://github.com/ayushap18/BlockArena
cd BlockArena

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

In another terminal, test the environment:

```bash
curl -X POST http://localhost:7860/reset
python3 test_environment.py
```

## Run Demo

```bash
python3 demo.py
```

## Run Inference Agent

```bash
export API_KEY=<your-openai-key>
export API_BASE_URL=https://api.openai.com/v1
export SERVER_URL=http://localhost:7860
python3 inference.py
```

## Project Structure

```
├── models.py                    # BlockArenaAction/Observation (Pydantic)
├── client.py                    # OpenEnv WebSocket client
├── __init__.py                  # Package exports
├── inference.py                 # LLM agent runner
├── demo.py                      # Interactive demo
├── test_environment.py          # Full test suite
├── Dockerfile                   # Docker container
├── pyproject.toml              # Package metadata
├── openenv.yaml                # OpenEnv manifest
└── server/
    ├── app.py                  # FastAPI server
    ├── blockarena_environment.py # Core environment
    ├── opponents.py            # Vendor & Legal agents
    ├── requirements.txt        # Server dependencies
    └── deals/
        ├── easy.json           # Easy tier deal
        ├── medium.json         # Medium tier deal
        └── hard.json           # Hard tier deal (3 probe budget)
```

## Environment Design

**Key Features:**
- Partially observable: agents cannot directly see opponent hidden values
- Deterministic: fully reproducible episodes
- Rule-based grader: no LLM in the reward loop
- Curriculum: easy → medium → hard progression
- Structured rewards: step rewards + episode bonuses

**Reward Structure:**
- ACCEPT/PROPOSE when both agree: +0.40
- Successful PROBE: +0.10
- Legal flag triggered: -0.20
- Vendor walkout: -0.30
- Final bonus (up to +0.40) for coherent strategy

## Testing

All tests pass:

```bash
python3 test_environment.py
```

Tests verify:
- ✓ Environment reset and initialization
- ✓ Probe action rewards
- ✓ Clause agreement mechanics
- ✓ Probe budget enforcement (hard tier)
- ✓ Episode determinism
- ✓ SUMMARIZE action (new feature)

## References

- **OpenEnv**: https://github.com/meta-pytorch/OpenEnv
- **GitHub**: https://github.com/ayushap18/BlockArena
- **HuggingFace Spaces**: https://huggingface.co/spaces/axy18/BlockArena

## License

MIT
