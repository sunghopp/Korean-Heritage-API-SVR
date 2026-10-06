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
tests preserve the original examples and instruction with 0, 1, and 5 previous
conversation turns, while allowing the deliberate additions: three shared social
examples before the original examples and a shared language policy after the
original instruction. Both fixed and RAG paths include these additions.
Do not regenerate the snapshot from a changed implementation just to pass tests.

Greeting tests check prompt construction, translation/reply separation, previous
conversation preservation, and the actual Gemini request builder with a mocked
model. RAG logs count retrieved pairs separately from the three shared social
examples. These tests cannot establish the real model's language consistency.

After deployment, reset the conversation and check these inputs on both the
fixed-example and RAG paths (confirm the selected path in the logs):

| Input | Expected behavior |
| --- | --- |
| 안녕하세요. 반갑습니다. | Short Jeju greeting; standard Korean translation retained |
| 예 반갑수다. 이번에 이디로 이사 와신디 전입신고 하젠 햄수다. | Jeju greeting and a relevant administrative response |
| 알았수다 고맙수다 | Jeju acknowledgment and closing |
| 감사합니다. 인터넷으로도 신청할 수 있나요? | Answer the follow-up rather than ending the conversation |
| A new question after an old standard Korean AI reply | Keep relevant context, but reply in Jeju |

Check both `ars_reply_text` on screen and spoken audio. Administrative facts must
still come from relevant source material; the social examples supply style only.

Before claiming the production issue is resolved, check the real 225 source
documents and captured retrieval responses. After deployment, look for
`프롬프트: 동적 Few-Shot N쌍`; rejected chunks log the filename, reason, section order,
parsed counts, character length, and at most 300 characters from each end.
Retrieval logs include the source URI and score. If sections are split across
chunks, this change will safely fall back, but retrieving a complete source
document or changing ingestion remains a separate decision.
