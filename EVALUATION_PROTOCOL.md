# Evaluation Protocol And Academic Framing

This file addresses the main evaluation risks for my university NLP assignment. I use it to make clear how the results should be interpreted.

## 1. Validation Set Size

The reported 100% scores are not my claim of general language understanding. They are controlled results on manually designed assignment test cases.

The validation is still useful because the cases cover the parts of the pipeline that matter for this design:

- greetings, goodbyes, help requests, quest requests, memory questions, compliments, insults, durable facts, and unknown filler for intent detection;
- name, lost item, helped person, weapon preference, search goal, insult, compliment, lost object, repair need, and fear memories for retrieval;
- cooperative, defiant, and mixed longer conversations for trace and trust behavior.

For a larger project, the next step would be to collect player-written utterances from multiple people and split them into development and held-out test sets.

## 2. Baselines

I now use two baseline levels:

1. A role-only LLM baseline, implemented in `baseline_agent.py`. This tests the assignment claim that a normal LLM prompt can roleplay Elara but does not provide external memory, trust state, retrieval metadata, or validation traces.
2. A keyword retrieval baseline, implemented in the notebooks and reusable in `evaluation_helpers.py`. This gives a measurable retrieval comparison against FAISS semantic search.

The keyword baseline is the quantitative baseline because it can be scored deterministically on the same memory retrieval cases. The role-only LLM baseline is mainly architectural: it demonstrates which capabilities disappear when the system is reduced to a single chat prompt.

## 3. Response Quality

Open-ended dialogue cannot be fully evaluated with one automatic metric. A response may be fluent but still ungrounded, or grounded but stylistically weak.

The project therefore combines:

- manual checklist review for character consistency, helpfulness, and tone;
- automatic checks in `evaluation_helpers.py` for trace completeness, trust arithmetic, and memory grounding;
- unsupported-memory prompts that check whether Elara avoids claiming memories that were not retrieved.

This is more informative than only reading the final chat output because it evaluates both final responses and the intermediate NLP pipeline decisions.

## 4. Model Choice

I use OpenAI models by default because they are easy to access through LangChain and stable for a short assignment demo. This is an engineering choice, not a conceptual dependency.

The architecture can use other models if equivalent LangChain chat and embedding wrappers are supplied, for example:

- a local Ollama chat model for response generation;
- a Hugging Face or sentence-transformers embedding model for vector retrieval;
- another hosted LLM provider supported by LangChain.

The `.env.example` file documents `OPENAI_MODEL` and `OPENAI_EMBEDDING_MODEL` so the OpenAI model names can be changed without editing code.

## 5. Memory Persistence

Memory is no longer only runtime state. I added JSON export/import so `MemoryNPC` can save and load:

- trust score;
- turn number;
- durable memories;
- conversation history;
- validation event log;
- pending advice state.

The Streamlit app exposes this as "Download memory state" and "Load memory state". When a state file is loaded, the backend rebuilds the FAISS index from stored memories.

This is still a simple persistence layer rather than a production database, but it is enough to demonstrate how the system could carry NPC memory across sessions.

## 6. Interpreting Perfect Scores

Several current validation results are 100%. These are acceptable only because the report presents them as controlled assignment checks. I avoid saying the system is "solved" or "perfect."

The correct interpretation of my results is:

> On manually designed validation cases, the pipeline met the target performance and behaved consistently enough to support the design claim.

Larger, user-generated, held-out validation would be needed before making broader claims.
