# MemoryNPC: A LangChain-Powered NPC Dialogue Agent with Long-Term Memory

MemoryNPC is a university NLP project about building functionality on top of an LLM. The demo lets a player talk to Elara, a cautious village blacksmith, but the main point is not the chat interface. The main point is the NLP pipeline: intent detection, memory extraction, vector storage, semantic retrieval, trust tracking, prompt construction, response generation, and validation.

## Design Challenge

Design a LangChain-based NPC memory agent to enable players in narrative game contexts to interact with non-player characters that remember previous conversations, retrieve relevant memories, and respond with context-aware dialogue with at least 80% memory retrieval success on manually designed test cases.

## Why This Is More Than A Normal Chatbot

A normal LLM can roleplay a blacksmith, but it is not a reliable long-term memory system by itself. It may forget details, hallucinate past facts, or hide its reasoning inside one large prompt. MemoryNPC adds inspectable components around the LLM:

- Intent labels show what the player is trying to do.
- Memory extraction decides what is durable enough to save.
- FAISS and OpenAI embeddings retrieve memories by meaning.
- A deterministic trust score tracks the relationship.
- The validation trace records what happened inside the pipeline.

This makes the system easier to test and explain.

## Assignment Coverage

This repository covers the assignment requirements as follows:

- Problem statement: stated in the design challenge format.
- Novel agent: an NPC memory agent that adds memory retrieval, trust state, and validation on top of an LLM.
- NLP pipeline: intent detection, memory extraction, embeddings, FAISS retrieval, prompt construction, and generation.
- Design motivation: documented in the notebook and summarized in this README.
- Validation: intent accuracy, retrieval success, keyword baseline comparison, trust score checks, response checklist, and app-side validation trace.
- Deliverables: notebook, code files, Streamlit demo, requirements, README, and environment example.

## Architecture Summary

The system uses a retrieval-augmented generation pipeline:

1. The player sends a message.
2. A LangChain intent detection step classifies the message.
3. A memory extraction step decides whether the message contains a durable memory.
4. Durable memories are stored as LangChain `Document` objects in a FAISS vector store using OpenAI embeddings.
5. Relevant memories are retrieved with semantic similarity search.
6. A deterministic trust score is updated from the detected intent.
7. A prompt template combines personality, trust score, trust level, intent, retrieved memories, recent conversation, and player input.
8. An OpenAI chat model generates Elara's response.

## Trust Score

Trust starts at `50` and is clamped between `0` and `100`.

Rules:

- `compliment`: `+5`
- `insult`: `-10`
- `share_fact`: `+1`
- `greeting`: `+1`
- `ask_help`, `ask_quest`, `ask_memory`, `goodbye`, `unknown`: `0`

Trust bands:

- Low trust: below `40`
- Neutral trust: `40-69`
- High trust: `70+`

The trust band is included in the response prompt. Low trust tells Elara to be more guarded and brief, neutral trust keeps her practical, and high trust makes her warmer and more helpful. The stored memory metadata also includes `trust_impact`, so relationship events can be inspected later.

Trust is not only changed by compliments and insults. Elara can also create a pending advice state after help or quest turns. On the next turn, if the player follows the advice, trust increases by `+3`. If the player explicitly does the opposite, such as saying they will ignore the advice or go alone anyway, trust decreases by `-8`. This makes trust depend on conversational behavior, not only sentiment.

## Validation Plan

The project validates the pipeline instead of only judging whether the final answer sounds nice.

- Intent detection evaluation: at least 20 test utterances with expected labels and accuracy calculation.
- Memory retrieval evaluation: at least 10 manually designed retrieval tests where the correct memory should appear in the top 3.
- Baseline comparison: a keyword-overlap retrieval baseline is compared with FAISS semantic retrieval.
- Trust validation: compliments and insults are checked against deterministic score changes and plotted.
- Advice-following validation: the validation lab checks whether following advice increases trust and ignoring advice decreases trust.
- Response quality checklist: manually checks relevant memory use, unsupported claims, trust tone, and character consistency.
- Streamlit validation trace: the app stores every turn with input, intent, extracted memory, saved memory, retrieved memories, trust before/after, trust delta, trust level, advice status, advice delta, and response. The trace can be downloaded as CSV.

The target is at least `80%` memory retrieval success on the manual retrieval test cases.

## Baseline

The main baseline is a normal LLM chatbot with only a role prompt. That baseline can generate plausible dialogue, but it does not expose a memory table, retrieval scores, trust metadata, or a validation trace.

The notebook also includes a keyword retrieval baseline. This checks whether a simple word-overlap method can retrieve the correct memory. FAISS should be better for semantically related queries, such as asking about a "weapon" when the stored memory says "sword."

## Files

- `npc_agent.py`: shared backend containing the `MemoryNPC` class.
- `app.py`: Streamlit chat demo and validation trace viewer.
- `notebook.ipynb`: report notebook with design motivation, implementation use, validation, results, error analysis, trustworthiness, and limitations.
- `validation_lab.ipynb`: technical validation notebook that works like a small notebook app and pressure-tests intent detection, memory extraction, retrieval, trust thresholds, unsupported memory claims, and trace completeness.
- `requirements.txt`: Python dependencies.
- `.env.example`: example environment variable file.
- `README.md`: setup and project explanation.

## Installation

```bash
cd "/Users/matthewmarinchev/Documents/SEM6-AI/CORE PROGRAM/NLP/memorynpc_project"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## API Key

Create a `.env` file in this folder:

```bash
OPENAI_API_KEY=your_api_key_here
```

Do not commit or share your real API key. The repository includes `.env.example`, not the real `.env`.

## Run The Notebook

Open `notebook.ipynb` in Jupyter or VS Code and run the cells from top to bottom. The notebook is the main report artifact. It explains what problem is being solved, why each component was chosen, how the architecture works, how the baseline is defined, and how validation is performed.

Open `validation_lab.ipynb` when you want a more technical testing notebook. It is less narrative and more focused on NLP pressure points. It includes a notebook-style `chat()` function, scripted conversations, trust-threshold response comparisons, retrieval-vs-keyword baseline tests, unsupported-memory checks, and validation-trace completeness checks.

## Run The Streamlit App

```bash
streamlit run app.py
```

The app keeps Elara alive in `st.session_state`, shows the chat, and displays trust score, trust level, stored memories, retrieved memories, and validation trace in the sidebar.

## Example Test Conversation

Try:

```text
Hello Elara
My name is Matt
I lost my sword near the forest
Do you remember what I lost?
You are a very skilled blacksmith
You are useless
Can you still help me?
Can you help me recover my sword?
I will go alone anyway
I will follow your advice and bring a torch
```

The validation trace should show the detected intent, saved memories, retrieved memories, trust changes, and Elara's response for each turn.

## Limitations

- The memory store is runtime-only unless extended with save/load logic.
- The prototype depends on the OpenAI API, so it needs internet access and an API key.
- The validation set is small and manually designed.
- The trust model is intentionally simple.
- The trust thresholds are prompt-level behavior controls, not hard game mechanics.
- The project is not connected to a real game engine.
- The system is designed for English dialogue only.
