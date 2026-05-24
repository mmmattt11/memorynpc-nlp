"""Streamlit demo for the MemoryNPC project.

The Streamlit app is intentionally simple. The assignment focus is the NLP
pipeline in npc_agent.py, so this file mostly provides:

1. a chat interface for trying the agent,
2. a sidebar that exposes internal state, and
3. a downloadable validation trace for testing/report evidence.
"""

import pandas as pd
import streamlit as st

from npc_agent import MemoryNPC


# Keep the page configuration minimal. A simple interface makes it easier for a
# teacher to inspect the NLP state instead of being distracted by UI complexity.
st.set_page_config(page_title="MemoryNPC Demo")

st.title("MemoryNPC Demo")
st.write(
    "Chat with Elara, a cautious village blacksmith. The demo uses LangChain, OpenAI, "
    "FAISS memory retrieval, intent detection, and deterministic trust tracking."
)


def create_agent() -> MemoryNPC:
    """Create one fresh MemoryNPC instance.

    This helper keeps construction in one place. It is used both at first page
    load and when the reset button is clicked.
    """
    return MemoryNPC()


def render_sidebar() -> None:
    """Render all inspectable state used for validation.

    The sidebar is not just decoration. It shows whether the pipeline is doing
    what the report claims: storing memories, retrieving memories, tracking
    trust, and logging each turn.
    """
    with st.sidebar:
        st.header("NPC State")

        if st.button("Reset conversation"):
            try:
                # Reset creates a new backend object instead of manually clearing
                # fields. This is less error-prone because MemoryNPC.reset() owns
                # the definition of a clean state.
                st.session_state.agent = create_agent()
                st.session_state.chat_rows = []
                st.session_state.error = None
                st.rerun()
            except Exception as exc:
                st.session_state.error = str(exc)

        if st.session_state.error:
            # Missing or invalid setup should be visible in the sidebar instead of
            # failing silently.
            st.error(st.session_state.error)
        elif st.session_state.agent:
            agent = st.session_state.agent
            # The numeric score is the deterministic relationship state.
            st.metric("Trust score", agent.trust_score)
            # The band explains how the numeric score affects dialogue tone.
            st.caption(f"Trust level: {agent.get_trust_level()} (<40 low, 40-69 neutral, 70+ high)")

            st.subheader("Stored memories")
            memory_table = agent.get_memory_table()
            if memory_table.empty:
                st.caption("No memories stored yet.")
            else:
                # The full table is useful for inspection and screenshots in a
                # report. It includes memory_id, type, importance, trust_impact,
                # and turn_number.
                st.dataframe(memory_table, width="stretch", hide_index=True)
                for memory in agent.memories:
                    # Streamlit dataframes can hide columns on narrow screens, so
                    # this plain-text summary makes trust_impact obvious.
                    impact = memory["trust_impact"]
                    impact_text = f"+{impact}" if impact > 0 else str(impact)
                    st.caption(f"Memory {memory['memory_id']}: {memory['text']} | trust_impact: {impact_text}")

            st.subheader("Last retrieved memories")
            if not agent.last_retrieved_memories:
                st.caption("No memories retrieved yet.")
            else:
                # Showing the last retrieval result helps verify that the final
                # LLM answer was grounded in actual retrieved memory.
                for memory in agent.last_retrieved_memories:
                    st.write(f"- {memory['text']}")

            st.subheader("Validation trace")
            event_log = agent.get_event_log()
            if event_log.empty:
                st.caption("No turns logged yet.")
            else:
                # The event log is the app-side version of the notebook validation
                # tables. It records intermediate NLP outputs for every turn.
                st.dataframe(event_log, width="stretch", hide_index=True)
                for event in agent.event_log:
                    st.caption(
                        f"Turn {event['turn']}: intent={event['intent']} | "
                        f"trust_delta={event['trust_delta']} | "
                        f"advice_status={event['advice_status']} | "
                        f"advice_delta={event['advice_delta']}"
                    )
                st.download_button(
                    "Download trace CSV",
                    event_log.to_csv(index=False),
                    file_name="memorynpc_validation_trace.csv",
                    mime="text/csv",
                )


if "agent" not in st.session_state:
    try:
        # Session state keeps the same MemoryNPC object alive across Streamlit
        # reruns. Without this, every message would start a new conversation and
        # the memory system could not be demonstrated.
        st.session_state.agent = create_agent()
        st.session_state.error = None
    except Exception as exc:
        # Store setup errors in session state so the UI can show a clear message
        # instead of a raw stack trace.
        st.session_state.agent = None
        st.session_state.error = str(exc)

if "chat_rows" not in st.session_state:
    # The visible chat is stored separately from the backend validation trace.
    # This keeps the UI history simple while npc_agent.py stores richer metadata.
    st.session_state.chat_rows = []

if st.session_state.error:
    render_sidebar()
    st.warning("Set `OPENAI_API_KEY` in a `.env` file or your environment, then reset the app.")
    st.stop()

agent = st.session_state.agent

# Re-render previous chat messages on every Streamlit rerun. Streamlit reruns the
# script after each interaction, so chat history must come from session_state.
for row in st.session_state.chat_rows:
    with st.chat_message(row["role"]):
        st.write(row["content"])

# Streamlit's chat input returns a value only after the user submits a message.
player_input = st.chat_input("Speak to Elara...")

if player_input:
    # Add the user message immediately so the interface feels like a chat app.
    st.session_state.chat_rows.append({"role": "user", "content": player_input})

    with st.chat_message("user"):
        st.write(player_input)

    with st.chat_message("assistant"):
        with st.spinner("Elara thinks back through her memories..."):
            # This is the only line that runs the full NLP pipeline. It returns
            # both Elara's response and the validation metadata for the turn.
            result = agent.generate_npc_response(player_input)
            st.write(result["npc_response"])

            with st.expander("Pipeline details"):
                # The expander provides a compact one-turn explanation for demos.
                # The full multi-turn trace is shown in the sidebar.
                details = pd.DataFrame(
                    [
                        {
                            "intent": result["intent"],
                            "saved_memory": result["saved_memory"] or "None",
                            "retrieved_memories": "; ".join(result["retrieved_memories"]) or "None",
                            "trust_score": result["trust_score"],
                            "trust_level": result["trust_level"],
                            "trust_delta": result["trust_delta"],
                            "advice_status": result["advice_status"],
                            "advice_delta": result["advice_delta"],
                            "new_advice": result["new_advice"],
                        }
                    ]
                )
                st.dataframe(details, width="stretch", hide_index=True)

    # Store the assistant message after generation so it appears on later reruns.
    st.session_state.chat_rows.append({"role": "assistant", "content": result["npc_response"]})

# Render the sidebar last so it reflects any state changes from the submitted
# message in the same rerun.
render_sidebar()
