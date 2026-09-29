# RAG regression tests

Run from the repository root in a Python 3.11+ virtual environment:

```sh
python -m pip install google-genai==2.13.0
python -B -m unittest discover -s tests -v
```

The SDK version matches `requirements.txt`. These tests use real Google GenAI
Content/Part objects but do not call a model, load GPU weights, or require cloud
credentials. Retrieval tests extract the real `retrieve_context` function with
AST to avoid `api_server`'s import-time model downloads; the RAG service is mocked.

The documents are synthetic fixtures following the v3 contract: three sections,
four Jeju/standard question pairs ending in `?`, and four aligned Jeju answers
ending in `다.` with no internal periods. A chunk that cannot supply all four
pairs is rejected as a whole, rather than partially paired. The parser supports
Markdown/plain/flattened/CRLF text and slashes inside either question.

`fixtures/pre_rag_prompt.json` freezes the system instruction and no-history
prompt for `이번 질문?` from commit `b15ac45`, before RAG was introduced. Fallback
tests compare against this snapshot with 0, 1, and 5 previous conversation turns.
Do not regenerate the snapshot from a changed implementation just to pass tests.

Before claiming the production issue is resolved, check the real 225 source
documents and captured retrieval responses. After deployment, look for
`프롬프트: 동적 Few-Shot N쌍`; rejected chunks log the filename, reason, section order,
parsed counts, character length, and at most 300 characters from each end.
Retrieval logs include the source URI and score. If sections are split across
chunks, this change will safely fall back, but retrieving a complete source
document or changing ingestion remains a separate decision.
