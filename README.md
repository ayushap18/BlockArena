# BlockArena

A strategic multi-party contract negotiation environment built for OpenEnv.

Built with precision for the OpenEnv Hackathon 2026.

## Team

Codecatalysts -- OpenEnv Hackathon 2026

Ayush -- Team Lead
Yatharth Gautam
Vardaan Dua

## What Is BlockArena?

BlockArena is a partially observable negotiation environment where an agent negotiates contract clauses against two opposing stakeholders:

- VendorAgent: protects business priorities and may walk out.
- LegalReviewer: enforces compliance redlines.

The objective is to close as many clauses as possible while maximizing episode-level score.

## What We Changed

This repository has been fully reworked and rebranded to BlockArena with updated naming, server identity, package metadata, and documentation.

## New Features Added

1. SUMMARIZE action

- New action type: SUMMARIZE
- Produces an in-episode negotiation summary in probe_result
- Does not consume negotiation rounds
- Helps agents checkpoint strategy before the next move

1. Strategy Guidance Metadata

- Every observation now includes:
  - risk_level: low | medium | high
  - next_best_action: context-aware action suggestion
  - action_mix: running count of actions used in the episode
- This gives agents a structured tactical signal beyond raw responses

## Action Space

- ACCEPT
- REJECT
- PROPOSE
- PROBE
- ESCALATE
- SUMMARIZE

## Observation Highlights

Each step returns:

- Current clause and clause text
- Vendor and legal responses
- Reward and done
- Metadata including:
  - vendor_stance
  - legal_stance
  - agreed_clauses
  - episode_score
  - probes_remaining
  - risk_level
  - next_best_action
  - action_mix

## Quick Start

```bash
git clone https://github.com/ayush18/BlockArena
cd BlockArena
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

In another terminal:

```bash
curl -X POST http://localhost:7860/reset
python test_environment.py
```

## Local Demo

```bash
python demo.py
```

## Run Inference Agent

```bash
export API_KEY=your_key
export API_BASE_URL=https://api.openai.com/v1
export SERVER_URL=http://localhost:7860
python inference.py
```

## Project Structure

- models.py: Action/Observation schemas
- client.py: OpenEnv client wrapper
- server/app.py: FastAPI OpenEnv app
- server/blockarena_environment.py: Core environment and rubric
- server/opponents.py: Vendor and legal behavior logic
- server/deals/: easy/medium/hard scenario configs

## License

MIT
