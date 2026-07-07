# Backend LLM Client

`backend.llm.LLMClient` is the shared model-calling boundary for backend agents.

Current implementation:

- uses the OpenAI Responses API
- loads `OPENAI_API_KEY` and `OPENAI_MODEL` from the repository-root `.env`
- defaults to `gpt-5.5`
- supports a simple text call through `complete_text(...)`
- has an early structured-output helper through `complete_json(...)`
- gives structured JSON generation a larger default output budget than simple text calls
- can opt into OpenAI Responses API web search with `web_search=True`
- can require web search with `web_search_required=True`
- stays synchronous until the backend has a real async runtime or async HTTP transport

Run the smoke test:

```bash
python3 backend/llm/client.py
```

The `.env` file is ignored by git. Do not commit API keys.
