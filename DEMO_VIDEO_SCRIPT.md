# Demo Video Script

Recommended length: 3 to 5 minutes.

## Recording Flow

1. Open with `memorynpc.ipynb` or the app folder and state the design challenge: an NPC memory agent for narrative game dialogue.
2. Start the app:

```bash
streamlit run app.py
```

3. In the app, send `Hello Elara`. Show the validation trace records the intent and trust state.
4. Send `My name is Matt`. Show the stored memory table.
5. Send `I lost my sword near the forest`. Show that the memory is saved.
6. Send `Do you remember what I lost?`. Show retrieved memories and Elara's grounded answer.
7. Send `You are a very skilled blacksmith`, then `You are useless`. Show trust changes.
8. Send `Can you help me recover my sword?`, then either `I will go alone anyway` or `I will follow your advice and bring a torch`. Show the advice-following status and trust delta.
9. Download the memory state from the sidebar.
10. Reset or reload the app, then load the memory state and show that memories and trace return.
11. End on the validation trace and explain that the notebooks evaluate intent accuracy, FAISS retrieval, keyword baseline, trust behavior, unsupported-memory behavior, and long conversations.

## What To Say

The important message is not that the chatbot sounds nice. The important message is that I built a university NLP project where the LLM is wrapped in an inspectable pipeline: intent detection, memory extraction, FAISS retrieval, deterministic trust state, persistence, response generation, and validation trace logging.
