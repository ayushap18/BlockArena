---
title: BlockArena
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
---

# 🎯 BlockArena

> A **strategic multi-party contract negotiation environment** for reinforcement learning research and AI development.

**Built with precision for the OpenEnv Hackathon 2026.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![OpenEnv](https://img.shields.io/badge/openenv-core-purple.svg)](https://github.com/meta-pytorch/OpenEnv)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

**Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/axy18/BlockArena) | **GitHub:** [ayushap18/BlockArena](https://github.com/ayushap18/BlockArena)

---

## 👥 Team

**Codecatalysts** — OpenEnv Hackathon 2026

| Role | Name |
|------|------|
| 🏆 Team Lead | **Ayush** |
| 🔧 Full-Stack | **Yatharth Gautam** |
| 🎨 Design & Strategy | **Vardaan Dua** |

---

## ❓ What Is BlockArena?

BlockArena is a **partially observable reinforcement learning environment** where an AI agent negotiates contract clauses against two opposing stakeholders:

- **💼 VendorAgent**: Protects business priorities and may walk out if pushed too hard
- **⚖️ LegalReviewer**: Enforces compliance redlines and flags problematic terms

**Objective:** Close as many clauses as possible while maximizing episode-level score through strategic probing and negotiation.

**Key Challenges:**
- Hidden opponent requirements (partial observability)
- Limited soft-skill resources (probes, rounds)
- Competing stakeholder interests
- Real-world complexity in contract terms

---

## ✨ Features

### 🎯 Feature 1: SUMMARIZE Action

Get a non-round-consuming negotiation checkpoint:

```python
action = BlockArenaAction(action_type="SUMMARIZE")
result = env.step(action)

print(result.observation.metadata['probe_result'])
# Output: "Progress: 2/4 clauses closed. Vendor favorable. Legal strict on IP."
```

**Benefits:**
- ✓ Helps agents checkpoint strategy without consuming rounds
- ✓ Returns detailed progress summary in `probe_result` 
- ✓ Useful for debugging and mid-episode reflection

### 🧠 Feature 2: Strategy Guidance Metadata

Every observation includes AI-powered tactical guidance:

```json
{
  "risk_level": "medium",
  "next_best_action": "PROPOSE",
  "action_mix": {
    "ACCEPT": 1,
    "REJECT": 0,
    "PROPOSE": 2,
    "PROBE": 3,
    "ESCALATE": 0,
    "SUMMARIZE": 0
  }
}
```

**Available Signals:**
- **risk_level** — `low | medium | high` — Context-aware risk assessment
- **next_best_action** — Recommended next action based on state
- **action_mix** — Running count of each action used this episode

---

## 🎮 Action Space

| Action | Icon | Purpose | Reward | Impact |
|--------|------|---------|--------|--------|
| **ACCEPT** | ✅ | Agree to current clause as-is | +0.40 (if both agree) | Closes clause |
| **REJECT** | ❌ | Reject clause with optional reason | Variable | Resets clause |
| **PROPOSE** | 💬 | Counter-propose new clause text | +0.40 (if both agree) | Negotiates terms |
| **PROBE** | 🔍 | Learn hidden requirements | +0.10 | Gains information |
| **ESCALATE** | ⚠️ | Force escalation (triggers legal flag) | -0.20 | High risk |
| **SUMMARIZE** | 📋 | Get negotiation progress summary | 0 (free) | No round cost |

---

## 📊 Difficulty Tiers

Choose your challenge level:

| Tier | Clauses | PROBE Budget | Rounds | Difficulty | Use Case |
|------|---------|-------------|--------|------------|----------|
| **Easy** 🟢 | 4 | Unlimited | 8 | Single hidden agenda | Learning basics |
| **Medium** 🟡 | 8 | Unlimited | 15 | Two opposing agendas | Standard training |
| **Hard** 🔴 | 12 | **3 only** | 20 | Conflicting requirements | Advanced agents |

---

## 🚀 Quick Start

### Local Setup (Recommended for Development)

#### 1. Clone & Install

```bash
# Clone repository
git clone https://github.com/ayushap18/BlockArena
cd BlockArena

# Install dependencies
pip install -r requirements.txt
```

#### 2. Start BlockArena Server

```bash
# Terminal 1: Run the FastAPI server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Once running, visit: **http://localhost:7860** 
You'll see:
```json
{
  "name": "BlockArena",
  "description": "Strategic Contract Negotiation Environment for OpenEnv",
  "team": "Codecatalysts - OpenEnv Hackathon 2026",
  "version": "1.0.0",
  "endpoints": {
    "reset": "POST /reset - Start a new negotiation episode",
    "step": "POST /step - Take an action in the current episode",
    "health": "GET /health - Health check",
    "docs": "GET /docs - API documentation (Swagger UI)",
    "openapi": "GET /openapi.json - OpenAPI schema"
  }
}
```

#### 3. Connect from Python

**Option A: Direct HTTP Client**

```python
import requests
import json

# Reset environment
reset_response = requests.post("http://localhost:7860/reset").json()
env_id = reset_response["env_id"]
print(f"Started episode: {env_id}")

# Take an action
action = {
    "action_type": "PROBE",
    "clause_id": "liability",
    "question": "What is the maximum liability cap?"
}

step_response = requests.post(
    "http://localhost:7860/step",
    json={"env_id": env_id, "action": action}
).json()

print(f"Reward: {step_response['reward']}")
print(f"Observation: {step_response['observation']}")
```

**Option B: OpenEnv SDK** (soon available)

```python
from blockarena import BlockArenaEnv, BlockArenaAction

# Connect to local server
env = BlockArenaEnv(base_url="http://localhost:7860")
obs = env.reset()

# Take action
action = BlockArenaAction(
    action_type="PROBE",
    clause_id="liability",
    question="Maximum liability cap?"
)
result = env.step(action)

print(f"Reward: {result.reward}")
print(f"Risk Level: {result.observation.metadata['risk_level']}")
print(f"Next Best Action: {result.observation.metadata['next_best_action']}")
```

#### 4. Run Interactive Demo

```bash
# Terminal 2: Run the demo agent
python3 demo.py
```

This shows a complete negotiation episode with visual output.

#### 5. Run AI Agent (Using LLM)

```bash
# Terminal 2: Set your API key and run inference agent
export API_KEY=your-openai-api-key
export API_BASE_URL=https://api.openai.com/v1
export SERVER_URL=http://localhost:7860

python3 inference.py
```

Watch a GPT-4 powered agent negotiate in real-time! 🤖

#### 6. Run Full Test Suite

```bash
# Verify everything works
python3 test_environment.py
```

Expected output:
```
✓ test_reset
✓ test_probe_reward
✓ test_propose_closes_clause
✓ test_probe_budget_hard
✓ test_deterministic
✓ test_summarize_action_does_not_consume_round
```

---

### 🌐 Use via HuggingFace Spaces

Simple and instant—no local setup needed!

Visit: **[BlockArena on HuggingFace Spaces](https://huggingface.co/spaces/axy18/BlockArena)**

The interactive playground includes:
- 📋 **Action Form**: Select action type, provide parameters
- 🎮 **Live Negotiation**: See real-time environment responses
- 📊 **State Inspector**: View full observation, rewards, metadata
- 🔄 **Reset**: Start new episodes on demand

---

### 📚 Connect Programmatically

#### From Any Environment

```python
import httpx
import asyncio

async def negotiate():
    base_url = "https://huggingface.co/spaces/axy18/BlockArena"  # Public Space endpoint
    
    # Reset
    async with httpx.AsyncClient() as client:
        reset = await client.post(f"{base_url}/reset")
        env_id = reset.json()["env_id"]
        
        # Step
        action = {
            "action_type": "PROPOSE",
            "clause_id": "liability",
            "new_text": "Maximum $1M liability cap",
            "reason": "Industry standard for contracts of this size"
        }
        
        step = await client.post(
            f"{base_url}/step",
            json={"env_id": env_id, "action": action}
        )
        
        result = step.json()
        print(f"✓ Episode reward: {result['reward']}")
        print(f"✓ Status: {result['done']}")

asyncio.run(negotiate())
```

---

## 🛠️ Development & Contribution

### Fork & Improve BlockArena

We welcome contributions! Here's how:

#### Step 1: Fork on OpenEnv Hub

```bash
openenv fork axy18/BlockArena --repo-id <your-username>/<your-repo-name>
cd <your-forked-repo>
```

#### Step 2: Make Your Changes

Examples of improvements:
- Add new opponent strategies
- Extend reward shaping
- Create new deal scenarios
- Improve observation richness
- Add new action types

```bash
# Edit files
git add .
git commit -m "feat: add new opponent strategy"
```

#### Step 3: Submit Pull Request

```bash
openenv push axy18/BlockArena --create-pr
```

Our team will review and merge! 🎉

**For GitHub contributions:**

```bash
git clone https://github.com/ayushap18/BlockArena
git checkout -b feature/your-feature
# Make changes, commit, push
# Open PR on GitHub
```

---

## 🧪 Advanced Usage

### Run Benchmarks

```bash
python3 inference.py --tier easy --model gpt-4
python3 inference.py --tier medium --model gpt-3.5-turbo
python3 inference.py --tier hard --model claude-3-opus
```

### Analyze Episodes

```python
from server.blockarena_environment import BlockArenaEnvironment
from blockarena import BlockArenaAction

env = BlockArenaEnvironment()
obs = env.reset()

# Check current state
print(f"Vendor Stance: {obs.metadata['vendor_stance']}")
print(f"Legal Risk: {obs.metadata['legal_risk_level']}")
print(f"Recommended: {obs.metadata['next_best_action']}")

# Use SUMMARIZE to get progress
result = env.step(BlockArenaAction(action_type="SUMMARIZE"))
print(f"\nProgress Summary:\n{result.observation.metadata['probe_result']}")
```

### Train Custom Agents

```bash
# The environment is RL-ready (gymnasium compatible)
python3 training_script.py --algorithm ppo --episodes 1000
```

---

## 📁 Project Structure

```
BlockArena/
├── 🎯 Core Environment
│   ├── models.py                    # Pydantic schemas
│   ├── client.py                    # OpenEnv WebSocket client
│   └── __init__.py                  # Package exports
│
├── 🌐 Server
│   ├── server/
│   │   ├── app.py                   # FastAPI app with /reset, /step routes
│   │   ├── blockarena_environment.py # Core RL environment logic
│   │   ├── opponents.py             # VendorAgent & LegalReviewer
│   │   ├── requirements.txt         # Server dependencies
│   │   └── deals/
│   │       ├── easy.json            # 4-clause deal
│   │       ├── medium.json          # 8-clause deal
│   │       └── hard.json            # 12-clause deal (3 probe limit)
│
├── 🤖 Agents & Demos
│   ├── inference.py                 # GPT-based agent runner
│   ├── demo.py                      # Interactive demo
│   └── test_environment.py          # Full test suite (6 tests)
│
├── 📦 Configuration
│   ├── Dockerfile                   # Docker container
│   ├── pyproject.toml              # Package metadata
│   ├── openenv.yaml                # OpenEnv manifest
│   ├── requirements.txt            # Root dependencies
│   └── uv.lock                     # Locked dependency versions
│
└── 📚 Documentation
    └── README.md                    # This file
```

---

## 🏗️ Environment Design

### Core Principles

**Partially Observable:** 
- Agents cannot directly see opponent hidden values (vendor max price, legal risk thresholds, etc.)
- Must use PROBE actions to gain information strategically

**Deterministic:**
- Fully reproducible episodes (save/load support)
- No randomness in opponent behavior (rule-based agents)
- Perfect for benchmarking and comparison

**Rule-Based Grading:**
- No LLM in the reward loop (fast, deterministic)
- Clear, interpretable reward signals
- Reward shaping for strategic behavior

**Curriculum Learning:**
- Easy tier: Single agent with clear agenda
- Medium tier: Two agents with different priorities
- Hard tier: Conflicting requirements + limited probes

### Reward Structure

```
┌─────────────────────────────────────────────┐
│ Episode Reward = Clause Rewards + Bonuses   │
├─────────────────────────────────────────────┤
│ ✓ ACCEPT/PROPOSE (both agree)   → +0.40    │
│ ✓ Successful PROBE               → +0.10    │
│ ✗ Legal flag triggered           → -0.20    │
│ ✗ Vendor walks out               → -0.30    │
│ ⭐ Final bonus (strategy quality) → +0.40    │
│                                  ───────────│
│ Total range: [-1.0, +4.8]        (clamped)  │
└─────────────────────────────────────────────┘
```

---

## 🧪 Testing

### Run All Tests

```bash
python3 test_environment.py
```

### Test Coverage

| Test | Purpose | Status |
|------|---------|--------|
| `test_reset` | Environment initialization | ✅ Pass |
| `test_probe_reward` | PROBE action rewards | ✅ Pass |
| `test_propose_closes_clause` | PROPOSE/ACCEPT agreement | ✅ Pass |
| `test_probe_budget_hard` | Hard tier probe limit | ✅ Pass |
| `test_deterministic` | Episode reproducibility | ✅ Pass |
| `test_summarize_action_does_not_consume_round` | SUMMARIZE feature | ✅ Pass |

**All 6 tests passing!** ✨

---

---

## 📖 References

| Resource | Link |
|----------|------|
| **OpenEnv Documentation** | [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv) |
| **GitHub Repository** | [github.com/ayushap18/BlockArena](https://github.com/ayushap18/BlockArena) |
| **HuggingFace Spaces** | [huggingface.co/spaces/axy18/BlockArena](https://huggingface.co/spaces/axy18/BlockArena) |
| **OpenEnv Hackathon 2026** | [OpenEnv.org](https://openenv.org) |

---

## ❓ FAQ

**Q: Can I use BlockArena without OpenEnv?**

A: Yes! You can interact with it as a standard FastAPI service via HTTP requests. No special dependencies needed.

**Q: What's the learning curve?**

A: Start with `demo.py` to see a full episode, then try the Easy tier with the Python client.

**Q: Can I add custom clauses?**

A: Absolutely! Edit the JSON files in `server/deals/` to create your own negotiation scenarios.

**Q: Is this suitable for training RL agents?**

A: Yes! The environment is gymnasium-compatible and works with standard RL libraries (PPO, DQN, etc.).

**Q: How do I report issues?**

A: Open an issue on [GitHub](https://github.com/ayushap18/BlockArena) or submit a PR with improvements!

---

## 🎁 What's Next?

Planned features for future iterations:

- 🤝 Multi-agent training benchmarks
- 📊 Episode analytics dashboard
- 🎨 Web UI for visualization
- 🔄 Continuous learning mode
- 📈 Leaderboard for agent performance
- 🌍 Internationalization (multilingual deals)

Interested in contributing? See **Development & Contribution** section above! 🚀

---

## License

MIT
