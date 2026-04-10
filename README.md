---
title: BlockArena
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# BlockArena

Strategic contract negotiation for OpenEnv, built for fast demos and RL experimentation.

**Live demo:** [Hugging Face Spaces](https://huggingface.co/spaces/axy18/BlockArena) | **GitHub:** [ayushap18/BlockArena](https://github.com/ayushap18/BlockArena)

## Highlights

- Partially observable negotiation with two opponents: VendorAgent and LegalReviewer
- Three difficulty tiers: easy, medium, hard
- New `SUMMARIZE` action for free progress checkpoints
- Strategy intelligence: `risk_level`, `next_best_action`, `negotiation_phase`, and `win_probability`
- Reward transparency via `reward_breakdown` metadata
- Polished Gradio cockpit with tier selector, guided demo, and benchmark leaderboard

## Quick Start

```bash
git clone https://github.com/ayushap18/BlockArena
cd BlockArena
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860` for the UI. API routes remain available at `/reset`, `/step`, `/info`, and `/health`.

## Run Checks

```bash
python3 test_environment.py
python3 inference.py
```

## Team

Codecatalysts for OpenEnv Hackathon 2026

- Ayush, Team Lead
- Yatharth Gautam
- Vardaan Dua

## Links

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- [BlockArena on Hugging Face](https://huggingface.co/spaces/axy18/BlockArena)
MIT
