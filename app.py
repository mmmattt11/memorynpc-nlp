import pandas as pd
import streamlit as st

from npc_agent import MemoryNPC


st.set_page_config(page_title="MemoryNPC Demo")

st.title("MemoryNPC Demo")
st.write(
    "Chat with Elara, a cautious village blacksmith. The demo uses LangChain, OpenAI, "
    "FAISS memory retrieval, intent detection, and deterministic trust tracking."
)


def create_agent() -> MemoryNPC:
    return MemoryNPC()


def render_sidebar() -> None:
    with st.sidebar:
        st.header("NPC State")

        if st.button("Reset conversation"):
            try:
                st.session_state.agent = create_agent()
                st.session_state.chat_rows = []
                st.session_state.error = None
                st.rerun()
            except Exception as exc:
                st.session_state.error = str(exc)

        if st.session_state.error:
            st.error(st.session_state.error)
        elif st.session_state.agent:
            agent = st.session_state.agent
            st.metric("Trust score", agent.trust_score)
            st.caption(f"Trust level: {agent.get_trust_level()} (<40 low, 40-69 neutral, 70+ high)")

            st.subheader("Stored memories")
            memory_table = agent.get_memory_table()
            if memory_table.empty:
                st.caption("No memories stored yet.")
            else:
                st.dataframe(memory_table, width="stretch", hide_index=True)
                for memory in agent.memories:
                    impact = memory["trust_impact"]
                    impact_text = f"+{impact}" if impact > 0 else str(impact)
                    st.caption(f"Memory {memory['memory_id']}: {memory['text']} | trust_impact: {impact_text}")

            st.subheader("Last retrieved memories")
            if not agent.last_retrieved_memories:
                st.caption("No memories retrieved yet.")
            else:
                for memory in agent.last_retrieved_memories:
                    st.write(f"- {memory['text']}")

            st.subheader("Validation trace")
            event_log = agent.get_event_log()
            if event_log.empty:
                st.caption("No turns logged yet.")
            else:
                st.dataframe(event_log, width="stretch", hide_index=True)
                st.download_button(
                    "Download trace CSV",
                    event_log.to_csv(index=False),
                    file_name="memorynpc_validation_trace.csv",
                    mime="text/csv",
                )


if "agent" not in st.session_state:
    try:
        st.session_state.agent = create_agent()
        st.session_state.error = None
    except Exception as exc:
        st.session_state.agent = None
        st.session_state.error = str(exc)

if "chat_rows" not in st.session_state:
    st.session_state.chat_rows = []

if st.session_state.error:
    render_sidebar()
    st.warning("Set `OPENAI_API_KEY` in a `.env` file or your environment, then reset the app.")
    st.stop()

agent = st.session_state.agent

for row in st.session_state.chat_rows:
    with st.chat_message(row["role"]):
        st.write(row["content"])

player_input = st.chat_input("Speak to Elara...")

if player_input:
    st.session_state.chat_rows.append({"role": "user", "content": player_input})

    with st.chat_message("user"):
        st.write(player_input)

    with st.chat_message("assistant"):
        with st.spinner("Elara thinks back through her memories..."):
            result = agent.generate_npc_response(player_input)
            st.write(result["npc_response"])

            with st.expander("Pipeline details"):
                details = pd.DataFrame(
                    [
                        {
                            "intent": result["intent"],
                            "saved_memory": result["saved_memory"] or "None",
                            "retrieved_memories": "; ".join(result["retrieved_memories"]) or "None",
                            "trust_score": result["trust_score"],
                            "trust_level": result["trust_level"],
                            "trust_delta": result["trust_delta"],
                        }
                    ]
                )
                st.dataframe(details, width="stretch", hide_index=True)

    st.session_state.chat_rows.append({"role": "assistant", "content": result["npc_response"]})

render_sidebar()
