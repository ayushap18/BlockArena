"""
BlockArena - Interactive RL Playground
Gradio-based dashboard for HuggingFace Spaces
"""

import gradio as gr
import json
from typing import Optional, Dict, Any, Tuple
from server.blockarena_environment import (
    BlockArenaEnvironment,
    BlockArenaAction,
    ActionType
)

# Global environment instance
env: Optional[BlockArenaEnvironment] = None
episode_history = []


def initialize_environment() -> str:
    """Initialize the environment."""
    global env
    try:
        env = BlockArenaEnvironment()
        obs = env.reset()
        episode_history.clear()
        return "✅ Environment initialized successfully!"
    except Exception as e:
        return f"❌ Error initializing environment: {str(e)}"


def get_initial_state() -> Tuple[str, str, str]:
    """Get the initial state when page loads."""
    if env is None:
        initialize_environment()
    
    if env is None:
        return "Not initialized", "N/A", "Error"
    
    obs = env.reset()
    state_json = json.dumps(obs.metadata, indent=2)
    return format_observation(obs), state_json, "Ready"


def reset_episode() -> Tuple[str, str, str, str]:
    """Reset the environment and start a new episode."""
    global env, episode_history
    
    if env is None:
        initialize_environment()
    
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
        
        return state_info, state_json, log_msg, "Ready to negotiate"
    except Exception as e:
        return f"❌ Reset failed: {str(e)}", "{}", "Error during reset", "Error"


def take_action(
    action_type: str,
    clause_id: str,
    new_text: Optional[str] = None,
    reason: Optional[str] = None,
    question: Optional[str] = None
) -> Tuple[str, str, str, str, str]:
    """Take an action in the environment."""
    global env, episode_history
    
    if env is None:
        return "❌ Environment not initialized", "{}", "Error", "Not initialized", ""
    
    if not action_type:
        return "⚠️ Please select an action type", "{}", "Warning", "Awaiting input", ""
    
    try:
        # Build action based on type
        action_kwargs = {"action_type": action_type}
        
        if action_type in ["ACCEPT", "REJECT", "PROPOSE", "PROBE"]:
            if not clause_id:
                return "⚠️ Clause ID is required for this action", "{}", "Warning", "Missing input", ""
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
        
        return state_info, state_json, action_log, status, action_response
        
    except Exception as e:
        error_msg = f"❌ Action failed: {str(e)}"
        return error_msg, "{}", f"Error: {str(e)}", "Error", ""


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
            gr.Markdown("### ⚙️ Environment Controls")
            reset_btn = gr.Button("🔄 Reset Episode", size="lg", variant="primary")
            
            tier_display = gr.Textbox(label="Current Tier", interactive=False, value="Not initialized")
            clauses_display = gr.Textbox(label="Total Clauses", interactive=False, value="0")
            
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Episode Status")
            state_display = gr.Textbox(
                label="Current State",
                interactive=False,
                lines=6,
                value="Click 'Reset Episode' to start"
            )
    
    gr.Markdown("---")
    
    # Action Panel
    gr.Markdown("### 🎮 Take Action")
    
    with gr.Row():
        with gr.Column(scale=1):
            action_type = gr.Dropdown(
                label="Action Type",
                choices=[
                    ("Accept Clause", "ACCEPT"),
                    ("Reject Clause", "REJECT"),
                    ("Propose Alternative", "PROPOSE"),
                    ("Probe for Info", "PROBE"),
                    ("Escalate", "ESCALATE"),
                    ("Summarize Progress", "SUMMARIZE")
                ],
                interactive=True,
                value=None
            )
            
            clause_id = gr.Textbox(
                label="Clause ID",
                placeholder="e.g., liability, payment_terms, ip_ownership",
                interactive=True
            )
        
        with gr.Column(scale=1):
            new_text = gr.Textbox(
                label="New Text (for PROPOSE)",
                placeholder="e.g., Maximum liability: $1M",
                interactive=True
            )
            
            reason = gr.Textbox(
                label="Reason (for REJECT)",
                placeholder="e.g., Exceeds our risk tolerance",
                interactive=True
            )
        
        with gr.Column(scale=1):
            question = gr.Textbox(
                label="Question (for PROBE)",
                placeholder="e.g., What is the vendor's liability expectation?",
                interactive=True,
                lines=2
            )
            
            step_btn = gr.Button("▶️ Execute Action", size="lg", variant="primary")
    
    gr.Markdown("---")
    
    # Results
    gr.Markdown("### 📈 Action Results")
    
    with gr.Row():
        with gr.Column(scale=1):
            action_result_display = gr.Markdown(
                value="Take an action to see results",
                label="Result"
            )
            
        with gr.Column(scale=1):
            metadata_display = gr.Code(
                language="json",
                label="Full Observation (JSON)",
                value="{}",
                interactive=False
            )
    
    gr.Markdown("---")
    
    # Bottom Row: Logs and Summary
    with gr.Row():
        with gr.Column():
            logs_display = gr.Textbox(
                label="📋 Action Log",
                interactive=False,
                lines=4,
                value="Ready to start"
            )
        
        with gr.Column():
            summary_display = gr.Markdown(
                value="No actions taken yet",
                label="Episode Summary"
            )
    
    # Status indicator
    status_display = gr.Textbox(
        label="Status",
        interactive=False,
        value="Ready",
    )
    
    gr.Markdown("---")
    
    # Footer
    gr.Markdown(
        """
        **BlockArena v1.0** | Built by [Codecatalysts](https://github.com/ayushap18/BlockArena)
        
        [GitHub](https://github.com/ayushap18/BlockArena) • [Paper](https://openenv.org)
        """
    )
    
    # ==================== EVENT HANDLERS ====================
    
    def on_reset():
        state_info, state_json, log_msg, status = reset_episode()
        summary = get_episode_summary()
        return (
            state_info,           # state_display
            state_json,           # metadata_display
            log_msg,              # logs_display
            summary,              # summary_display
            status,               # status_display
        )
    
    def on_action(atype, cid, ntext, reason_val, question_val):
        state_info, state_json, log_msg, status, action_resp = take_action(
            atype, cid, ntext, reason_val, question_val
        )
        summary = get_episode_summary()
        return (
            state_info,           # state_display
            state_json,           # metadata_display
            action_resp,          # action_result_display
            log_msg,              # logs_display
            summary,              # summary_display
            status,               # status_display
        )
    
    # Wire up button clicks
    reset_btn.click(
        on_reset,
        outputs=[
            state_display,
            metadata_display,
            logs_display,
            summary_display,
            status_display
        ]
    )
    
    step_btn.click(
        on_action,
        inputs=[action_type, clause_id, new_text, reason, question],
        outputs=[
            state_display,
            metadata_display,
            action_result_display,
            logs_display,
            summary_display,
            status_display
        ]
    )
    
    # Initialize on load
    demo.load(
        get_initial_state,
        outputs=[state_display, metadata_display, status_display]
    )


if __name__ == "__main__":
    initialize_environment()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
