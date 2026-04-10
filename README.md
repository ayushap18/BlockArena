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
- Strategy metadata: `risk_level`, `next_best_action`, and `action_mix`
- Polished Gradio cockpit with tier selector, live metrics, and guided demo

## Quick Start

```bash
git clone https://github.com/ayushap18/BlockArena
cd BlockArena
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860` for the API or `http://localhost:7860/playground` for the UI.

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
