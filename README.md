# MemoryNPC: A LangChain-Powered NPC Dialogue Agent with Long-Term Memory

MemoryNPC is my university NLP assignment project about building functionality on top of an LLM. I designed the project myself around the assignment brief. The demo lets a player talk to Elara, a cautious village blacksmith, but the main point is not the chat interface. The main point is the NLP pipeline: intent detection, memory extraction, vector storage, semantic retrieval, trust tracking, prompt construction, response generation, persistence, and validation.

## Design Challenge

Design a LangChain-based NPC memory agent to enable players in narrative game contexts to interact with non-player characters that remember previous conversations, retrieve relevant memories, and respond with context-aware dialogue with at least 80% memory retrieval success on manually designed test cases.

## Why This Is More Than A Normal Chatbot

A normal LLM can roleplay a blacksmith, but it is not a reliable long-term memory system by itself. It may forget details, hallucinate past facts, or hide its reasoning inside one large prompt. MemoryNPC adds inspectable components around the LLM:

- Intent labels show what the player is trying to do.
- Memory extraction decides what is durable enough to save.
- FAISS and OpenAI embeddings retrieve memories by meaning.
- A deterministic trust score tracks the relationship.
- JSON state export/import can persist memory between sessions.
- The validation trace records what happened inside the pipeline.

This makes the system easier to test and explain.

## Assignment Coverage

I cover the assignment requirements as follows:

- Problem statement: stated in the design challenge format.
- Novel agent: an NPC memory agent that adds memory retrieval, trust state, and validation on top of an LLM.
- NLP pipeline: intent detection, memory extraction, embeddings, FAISS retrieval, prompt construction, and generation.
- Design motivation: documented in the notebook and summarized in this README.
- Validation: intent accuracy, retrieval success, keyword baseline comparison, trust score checks, response checklist, and app-side validation trace.

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
9. Optional JSON state export/import persists memories, trust, conversation history, and validation trace across sessions.

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

I validate the pipeline instead of only judging whether the final answer sounds nice. The goal is to make the design choices and failure risks visible.

- Intent detection evaluation: at least 20 test utterances with expected labels and accuracy calculation.
- Memory retrieval evaluation: at least 10 manually designed retrieval tests where the correct memory should appear in the top 3.
- Baseline comparison: a keyword-overlap retrieval baseline is compared with FAISS semantic retrieval.
- Trust validation: compliments and insults are checked against deterministic score changes and plotted.
- Advice-following validation: the validation lab checks whether following advice increases trust and ignoring advice decreases trust.
- Response quality checklist: manually checks relevant memory use, unsupported claims, trust tone, and character consistency.
- Automatic response checks: reusable helper functions check trace completeness, trust arithmetic, and simple memory-grounding conditions.
- Streamlit validation trace: the app stores every turn with input, intent, extracted memory, saved memory, retrieved memories, trust before/after, trust delta, trust level, advice status, advice delta, and response. The trace can be downloaded as CSV.

The target is at least `80%` memory retrieval success on the manual retrieval test cases.

## Baseline

The main architectural baseline is a normal LLM chatbot with only a role prompt. It is implemented in `baseline_agent.py`. That baseline can generate plausible dialogue, but it does not expose a memory table, retrieval scores, trust metadata, persistence, or a validation trace.

The notebook also includes a keyword retrieval baseline. This checks whether a simple word-overlap method can retrieve the correct memory. FAISS should be better for semantically related queries, such as asking about a "weapon" when the stored memory says "sword."

## Files

- `memorynpc.ipynb`: main notebook with design, architecture, validation summary, limitations, and code appendices.
- `memorynpc.html`: exported HTML version of the main notebook.
- `validation.ipynb`: executed validation notebook with at least 20 cases per major metric.
- `validation.html`: exported HTML version of the validation notebook.
- `run_validation.py`: reproducible validation script that writes CSV/JSON result files.
- `npc_agent.py`: shared backend containing the `MemoryNPC` class.
- `baseline_agent.py`: role-only LLM baseline used to explain what the system adds beyond a normal chatbot.
- `evaluation_helpers.py`: reusable baseline and automatic validation helpers.
- `app.py`: Streamlit chat demo and validation trace viewer.
- `VALIDATION_RESULTS.md`: concise summary of the executed notebook results and long-conversation validation outcomes.
- `EVALUATION_PROTOCOL.md`: explanation of validation strength, baselines, model choice, persistence, and how to interpret the high scores.
- `DEMO_VIDEO_SCRIPT.md`: short recording plan for the optional demo video.
- `requirements.txt`: Python dependencies.
- `.env.example`: example environment variable file.
- `README.md`: setup and project explanation.

## Installation

```bash
git clone <repository-url>
cd memorynpc_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## API Key

Create a `.env` file in this folder:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Do not commit or share your real API key. The repository includes `.env.example`, not the real `.env`.

I use OpenAI models by default because they are easy to run through LangChain for this assignment. The architecture is not limited to OpenAI: the same design can use local Ollama chat models, Hugging Face or sentence-transformers embeddings, or other LangChain-compatible model providers if the chat and embedding wrappers are swapped.

## Run The Notebook

Open `memorynpc.ipynb` first. It contains the complete project explanation and code appendices.

Open `validation.ipynb` for the detailed validation report. It tests every major metric at least 20 times and stores the detailed results in `results/validation/`.

The exported HTML versions are `memorynpc.html` and `validation.html`.

## Run The Streamlit App

```bash
streamlit run app.py
```

The app keeps Elara alive in `st.session_state`, shows the chat, and displays trust score, trust level, stored memories, retrieved memories, and validation trace in the sidebar. The sidebar also supports downloading and loading JSON memory state, so the system can persist Elara's memory across sessions.

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

- The JSON persistence layer is simple. A real game would likely use a database or game-save integration.
- The default configuration depends on the OpenAI API, so it needs internet access and an API key. Other LangChain-compatible chat and embedding models could be substituted.
- The validation set is small and manually designed. The 100% results should be interpreted as controlled assignment results, not general performance on all possible player language.
- The trust model is intentionally simple.
- The trust thresholds are prompt-level behavior controls, not hard game mechanics.
- The project is not connected to a real game engine.
- The system is designed for English dialogue only.
