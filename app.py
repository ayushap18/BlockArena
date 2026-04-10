"""
BlockArena - Interactive RL Playground
FastAPI/OpenEnv app with a mounted Gradio dashboard.
"""

import json
from typing import Optional, Dict, Any, Tuple

import gradio as gr

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. "
        "Install dependencies and ensure the OpenEnv package is available."
    ) from e

try:
    from models import BlockArenaAction, BlockArenaObservation
    from server.blockarena_environment import BlockArenaEnvironment
except ImportError:
    from models import BlockArenaAction, BlockArenaObservation
    from server.blockarena_environment import BlockArenaEnvironment

# Global environment instance
env: Optional[BlockArenaEnvironment] = None
episode_history = []
current_tier = "easy"

TIER_PROFILES = {
    "easy": {
        "clauses": 4,
        "probes": "Unlimited",
        "rounds": 8,
        "label": "Onboarding",
        "summary": "Best for first-time judges. Fast agreement path, clear signals, low risk.",
    },
    "medium": {
        "clauses": 8,
        "probes": "Unlimited",
        "rounds": 15,
        "label": "Balanced",
        "summary": "Shows negotiation depth with enough complexity to demonstrate reasoning.",
    },
    "hard": {
        "clauses": 12,
        "probes": 3,
        "rounds": 20,
        "label": "Showcase",
        "summary": "High-pressure mode for demonstrating robust strategy and resource management.",
    },
}


def get_tier_profile(tier: str) -> Dict[str, Any]:
    return TIER_PROFILES.get(tier, TIER_PROFILES["easy"])


def render_scenario_preview(tier: str) -> str:
    profile = get_tier_profile(tier)
    return f"""
### Scenario Preview

**Tier:** {tier.title()} ({profile['label']})  
**Clauses:** {profile['clauses']}  
**Probe Budget:** {profile['probes']}  
**Rounds:** {profile['rounds']}  

**Why this matters:** {profile['summary']}
"""


def render_live_metrics(obs) -> str:
    meta = obs.metadata
    return f"""
### Live Strategy Snapshot

| Metric | Value |
|---|---:|
| Clauses Agreed | {meta.get('clauses_agreed', 0)}/{meta.get('clauses_total', 0)} |
| Rounds Remaining | {obs.rounds_remaining} |
| Probes Remaining | {meta.get('probes_remaining', 0)} |
| Risk Level | {meta.get('risk_level', 'N/A')} |
| Next Best Action | {meta.get('next_best_action', 'N/A')} |
| Episode Score | {meta.get('episode_score', 0.01):.4f} |
"""


def render_action_history() -> str:
    if not episode_history:
        return "### Action Timeline\n\nNo actions taken yet."

    lines = ["### Action Timeline"]
    for index, action in enumerate(episode_history[-10:], start=max(len(episode_history) - 9, 1)):
        lines.append(
            f"{index}. **{action['action']}** on `{action['clause_id'] or 'n/a'}` "
            f"→ reward `{action['reward']:+.3f}` | done `{action['done']}`"
        )
    return "\n".join(lines)


def render_episode_summary() -> str:
    if not episode_history:
        return "### Episode Summary\n\nNo actions taken yet."

    total_reward = sum(a["reward"] for a in episode_history)
    action_counts = {}
    for action in episode_history:
        action_counts[action["action"]] = action_counts.get(action["action"], 0) + 1

    counts_text = "\n".join(f"- {atype}: {count}" for atype, count in sorted(action_counts.items()))
    return f"""
### Episode Summary

**Total Steps:** {len(episode_history)}  
**Total Reward:** {total_reward:+.3f}

**Action Mix**
{counts_text}
"""


def initialize_environment(tier: str = "easy") -> str:
    """Initialize the environment."""
    global env, current_tier
    try:
        current_tier = tier
        env = BlockArenaEnvironment(tier)
        obs = env.reset()
        episode_history.clear()
        return "✅ Environment initialized successfully!"
    except Exception as e:
        return f"❌ Error initializing environment: {str(e)}"


def get_initial_state() -> Tuple[str, str, str]:
    """Get the initial state when page loads."""
    if env is None:
        initialize_environment(current_tier)
    
    if env is None:
        return "Not initialized", "N/A", "Error"
    
    obs = env.reset()
    state_json = json.dumps(obs.metadata, indent=2)
    return format_observation(obs), state_json, "Ready"


def reset_episode(tier: Optional[str] = None) -> Tuple[str, str, str, str, str, str, str]:
    """Reset the environment and start a new episode."""
    global env, episode_history
    
    selected_tier = tier or current_tier
    initialize_environment(selected_tier)
    
    try:
        obs = env.reset()
        episode_history.clear()
        
        state_info = f"""
        🔄 **Episode Reset**
        
        **Tier:** {obs.metadata.get('tier', 'Unknown')}
        **Total Clauses:** {obs.metadata.get('total_clauses', 0)}
        **Starting Stance:** Vendor={obs.metadata.get('vendor_stance', 'N/A')}, Legal={obs.metadata.get('legal_risk_level', 'N/A')}
        """
        
        state_json = json.dumps(obs.metadata, indent=2)
        log_msg = "📋 New episode started"
        
        return (
            state_info,
            state_json,
            log_msg,
            "Ready to negotiate",
            render_scenario_preview(selected_tier),
            render_live_metrics(obs),
            render_action_history(),
        )
    except Exception as e:
        return (
            f"❌ Reset failed: {str(e)}",
            "{}",
            "Error during reset",
            "Error",
            render_scenario_preview(selected_tier),
            "### Live Strategy Snapshot\n\nUnavailable.",
            render_action_history(),
        )


def take_action(
    action_type: str,
    clause_id: str,
    new_text: Optional[str] = None,
    reason: Optional[str] = None,
    question: Optional[str] = None
) -> Tuple[str, str, str, str, str, str, str]:
    """Take an action in the environment."""
    global env, episode_history
    
    if env is None:
        return "❌ Environment not initialized", "{}", "Error", "Not initialized", "", "", ""
    
    if not action_type:
        return "⚠️ Please select an action type", "{}", "Warning", "Awaiting input", "", "", ""
    
    try:
        # Build action based on type
        action_kwargs = {"action_type": action_type}
        
        if action_type in ["ACCEPT", "REJECT", "PROPOSE", "PROBE"]:
            if not clause_id:
                return "⚠️ Clause ID is required for this action", "{}", "Warning", "Missing input", "", "", ""
            action_kwargs["clause_id"] = clause_id
        
        if action_type == "PROPOSE" and new_text:
            action_kwargs["new_text"] = new_text
        
        if action_type == "REJECT" and reason:
            action_kwargs["reason"] = reason
        
        if action_type == "PROBE" and question:
            action_kwargs["question"] = question
        
        if action_type == "SUMMARIZE":
            # SUMMARIZE doesn't need additional fields
            pass
        
        action = BlockArenaAction(**action_kwargs)
        result = env.step(action)
        
        # Log action
        action_log = f"Taking {action_type} on {clause_id if clause_id else 'N/A'}"
        episode_history.append({
            "action": action_type,
            "clause_id": clause_id,
            "reward": float(result.reward),
            "done": result.done
        })
        
        # Format response
        obs = result.observation
        reward = float(result.reward)
        done = result.done
        
        state_info = f"""
        🎯 **Action Result**
        
        **Reward:** {reward:+.3f}
        **Done:** {done}
        **Rounds Used:** {obs.metadata.get('rounds_used', 0)}
        **Risk Level:** {obs.metadata.get('risk_level', 'N/A')}
        **Suggested Next Action:** {obs.metadata.get('next_best_action', 'N/A')}
        **Probe Result:** {obs.metadata.get('probe_result', 'N/A')}
        """
        
        state_json = json.dumps(obs.metadata, indent=2)
        
        # Build action response
        action_response = f"""
        ✅ **{action_type} Executed**
        
        Reward: `{reward:+.3f}`
        Episode Done: `{done}`
        """
        
        status = "✅ Episode Complete!" if done else "▶️ Continuing..."
        
        return (
            state_info,
            state_json,
            action_log,
            status,
            action_response,
            render_live_metrics(obs),
            render_action_history(),
        )
        
    except Exception as e:
        error_msg = f"❌ Action failed: {str(e)}"
        return error_msg, "{}", f"Error: {str(e)}", "Error", "", "", render_action_history()


def run_guided_demo(tier: Optional[str] = None) -> Tuple[str, str, str, str, str, str, str]:
    """Run a short showcase episode that demonstrates the product quickly."""
    global env

    selected_tier = tier or current_tier
    initialize_environment(selected_tier)

    try:
        obs = env.reset()
        episode_history.clear()

        vendor_value = env._deal["vendor_hidden"]["value"]
        scripted_actions = [
            {"action_type": "PROBE", "clause_id": obs.clause_id, "party": "vendor", "question": "What matters most in this clause?"},
            {
                "action_type": "PROPOSE",
                "clause_id": obs.clause_id,
                "new_text": f"Negotiated terms for {obs.clause_id}: {vendor_value} included.",
            },
            {"action_type": "SUMMARIZE", "clause_id": obs.clause_id},
        ]

        last_result = None
        for action_kwargs in scripted_actions:
            action = BlockArenaAction(**action_kwargs)
            last_result = env.step(action)
            episode_history.append(
                {
                    "action": action_kwargs["action_type"],
                    "clause_id": action_kwargs.get("clause_id", ""),
                    "reward": float(last_result.reward),
                    "done": last_result.done,
                }
            )
            obs = last_result

        state_info = f"""
        🎬 **Guided Demo Complete**

        **Tier:** {selected_tier.title()}
        **Current Clause:** {obs.clause_id}
        **Vendor Response:** {obs.vendor_response}
        **Legal Response:** {obs.legal_response}
        """
        state_json = json.dumps(obs.metadata, indent=2)
        action_response = """
        ✅ **Demo Walkthrough Finished**

        This path shows the probe + summarize workflow judges can follow immediately.
        """
        return (
            state_info,
            state_json,
            "🚀 Guided demo executed",
            action_response,
            render_live_metrics(obs),
            render_action_history(),
            render_episode_summary(),
        )
    except Exception as e:
        return (
            f"❌ Guided demo failed: {str(e)}",
            "{}",
            "Error",
            "",
            "### Live Strategy Snapshot\n\nUnavailable.",
            render_action_history(),
            render_episode_summary(),
        )


def format_observation(obs) -> str:
    """Format observation for display."""
    metadata = obs.metadata
    return f"""
    📊 **Current State**
    
    **Vendor Stance:** {metadata.get('vendor_stance', 'N/A')}
    **Legal Risk:** {metadata.get('legal_risk_level', 'N/A')}
    **Rounds Used:** {metadata.get('rounds_used', 0)}
    **Risk Level:** {metadata.get('risk_level', 'N/A')}
    **Next Best Action:** {metadata.get('next_best_action', 'N/A')}
    """


def get_episode_summary() -> str:
    """Get a summary of the current episode."""
    if not episode_history:
        return "No actions taken yet"
    
    total_reward = sum(a["reward"] for a in episode_history)
    action_counts = {}
    for action in episode_history:
        atype = action["action"]
        action_counts[atype] = action_counts.get(atype, 0) + 1
    
    summary = f"""
    📈 **Episode Summary**
    
    **Total Steps:** {len(episode_history)}
    **Total Reward:** {total_reward:+.3f}
    **Actions Taken:**
    """
    
    for atype, count in sorted(action_counts.items()):
        summary += f"\n- {atype}: {count}"
    
    return summary


# ==================== GRADIO INTERFACE ====================

with gr.Blocks(
    title="BlockArena - RL Playground",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"),
    css="""
    .header-title { font-size: 2.5em; font-weight: bold; color: #4f46e5; }
    .action-panel { border: 2px solid #e5e7eb; border-radius: 8px; padding: 16px; background: #f9fafb; }
    .status-box { border-radius: 8px; padding: 12px; margin: 8px 0; }
    .status-success { background: #dcfce7; color: #166534; border-left: 4px solid #22c55e; }
    .status-warning { background: #fef3c7; color: #92400e; border-left: 4px solid #f59e0b; }
    .status-error { background: #fee2e2; color: #991b1b; border-left: 4px solid #ef4444; }
    """
) as demo:
    
    # Header
    gr.Markdown(
        """
        # 🎯 BlockArena - Strategic Contract Negotiation RL Playground
        
        **An OpenEnv environment for negotiation strategy research.**
        
        Negotiate complex multi-party contracts against AI opponents (VendorAgent & LegalReviewer).
        Learn to maximize rewards through strategic probing, proposing, and escalation.
        
        ---
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Mission Control")
            tier_selector = gr.Dropdown(
                label="Scenario Tier",
                choices=[("Easy", "easy"), ("Medium", "medium"), ("Hard", "hard")],
                value=current_tier,
                interactive=True,
            )
            reset_btn = gr.Button("🔄 Reset Episode", size="lg", variant="primary")
            guided_demo_btn = gr.Button("🎬 Run Guided Demo", size="lg", variant="secondary")
            scenario_preview = gr.Markdown(render_scenario_preview(current_tier))

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Live Negotiation Board")
            state_display = gr.Markdown(value="Click **Reset Episode** to load the selected scenario.")
            metrics_display = gr.Markdown("### Live Strategy Snapshot\n\nReady.")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎮 Action Panel")
            action_type = gr.Dropdown(
                label="Action Type",
                choices=[
                    ("Accept Clause", "ACCEPT"),
                    ("Reject Clause", "REJECT"),
                    ("Propose Alternative", "PROPOSE"),
                    ("Probe for Info", "PROBE"),
                    ("Escalate", "ESCALATE"),
                    ("Summarize Progress", "SUMMARIZE"),
                ],
                interactive=True,
                value=None,
            )
            clause_id = gr.Textbox(
                label="Clause ID",
                placeholder="e.g., liability, payment_terms, ip_ownership",
                interactive=True,
            )
            new_text = gr.Textbox(
                label="New Text (for PROPOSE)",
                placeholder="e.g., Maximum liability: $1M",
                interactive=True,
            )
            reason = gr.Textbox(
                label="Reason (for REJECT)",
                placeholder="e.g., Exceeds our risk tolerance",
                interactive=True,
            )
            question = gr.Textbox(
                label="Question (for PROBE)",
                placeholder="e.g., What matters most in this clause?",
                interactive=True,
                lines=2,
            )
            step_btn = gr.Button("▶️ Execute Action", size="lg", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 🧭 Results & Insights")
            action_result_display = gr.Markdown(value="Take an action to see results.")
            status_display = gr.Textbox(label="Status", interactive=False, value="Ready")
            logs_display = gr.Markdown("### Action Timeline\n\nNo actions taken yet.")
            summary_display = gr.Markdown("### Episode Summary\n\nNo actions taken yet.")

    gr.Markdown("### 🔍 Debug View")
    metadata_display = gr.Code(
        language="json",
        label="Full Observation (JSON)",
        value="{}",
        interactive=False,
    )

    gr.Markdown(
        """
        **BlockArena v1.0** | Built by [Codecatalysts](https://github.com/ayushap18/BlockArena)

        [GitHub](https://github.com/ayushap18/BlockArena) • [OpenEnv](https://openenv.org)
        """
    )

    # ==================== EVENT HANDLERS ====================

    def on_tier_change(tier):
        return render_scenario_preview(tier)

    def on_reset(tier):
        state_info, state_json, log_msg, status, scenario_text, metrics_text, history_text = reset_episode(tier)
        summary_text = render_episode_summary()
        return (
            state_info,
            state_json,
            metrics_text,
            history_text,
            summary_text,
            log_msg,
            status,
        )

    def on_action(atype, cid, ntext, reason_val, question_val):
        state_info, state_json, log_msg, status, action_resp, metrics_text, history_text = take_action(
            atype, cid, ntext, reason_val, question_val
        )
        summary_text = render_episode_summary()
        return (
            state_info,
            state_json,
            action_resp,
            metrics_text,
            history_text,
            summary_text,
            status,
        )

    def on_guided_demo(tier):
        state_info, state_json, log_msg, action_resp, metrics_text, history_text, summary_text = run_guided_demo(tier)
        return (
            state_info,
            state_json,
            action_resp,
            metrics_text,
            history_text,
            summary_text,
            "✅ Guided demo complete",
        )

    tier_selector.change(on_tier_change, inputs=[tier_selector], outputs=[scenario_preview])

    reset_btn.click(
        on_reset,
        inputs=[tier_selector],
        outputs=[state_display, metadata_display, metrics_display, logs_display, summary_display, action_result_display, status_display],
    )

    step_btn.click(
        on_action,
        inputs=[action_type, clause_id, new_text, reason, question],
        outputs=[state_display, metadata_display, action_result_display, metrics_display, logs_display, summary_display, status_display],
    )

    guided_demo_btn.click(
        on_guided_demo,
        inputs=[tier_selector],
        outputs=[state_display, metadata_display, action_result_display, metrics_display, logs_display, summary_display, status_display],
    )

    demo.load(
        get_initial_state,
        outputs=[state_display, metadata_display, status_display],
    )


app = create_app(
    BlockArenaEnvironment,
    BlockArenaAction,
    BlockArenaObservation,
    env_name="blockarena",
    max_concurrent_envs=1,
)


@app.get("/info")
async def info():
    """Environment metadata and endpoint reference."""
    return {
        "name": "BlockArena",
        "description": "Strategic Contract Negotiation Environment for OpenEnv",
        "team": "Codecatalysts - OpenEnv Hackathon 2026",
        "version": "1.0.0",
        "playground": "/",
        "endpoints": {
            "reset": "POST /reset - Start a new negotiation episode",
            "step": "POST /step - Take an action in the current episode",
            "health": "GET /health - Health check",
            "docs": "GET /docs - API documentation (Swagger UI)",
            "openapi": "GET /openapi.json - OpenAPI schema",
            "playground": "GET / - Interactive dashboard",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "blockarena",
        "ready": True,
    }


app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
