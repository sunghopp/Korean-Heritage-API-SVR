"""Check the actual Gemini request builder without cloud calls/model loading."""
import ast
import logging
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from google.genai import types
from pydantic import BaseModel, Field

from ars_prompt import RAG_SYSTEM_INSTRUCTION, SOCIAL_FEW_SHOT_CASES, build_prompt
from test_ars_prompt import document, serialize, SOCIAL_TURNS, SOURCE


class GeminiRequestTests(unittest.TestCase):
    def test_request_and_log_distinguish_rag_from_shared_examples(self):
        source = Path(__file__).resolve().parents[1] / "api_server.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if (
            isinstance(n, (ast.ClassDef, ast.FunctionDef))
            and n.name in {"GeminiARSResult", "ConversationTurn", "call_gemini_ars"}
        )]
        for refs in ([], [(SOURCE, document())]):
            with self.subTest(rag=bool(refs)):
                generate = Mock()
                env = {
                    "BaseModel": BaseModel, "Field": Field, "types": types, "math": math,
                    "RAG_SYSTEM_INSTRUCTION": RAG_SYSTEM_INSTRUCTION,
                    "SOCIAL_FEW_SHOT_CASES": SOCIAL_FEW_SHOT_CASES,
                    "build_prompt": build_prompt, "retrieve_context": Mock(return_value=refs),
                    "gemini_client": SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
                    "GEMINI_TUNED_ENDPOINT": "test-endpoint", "logger": logging.getLogger("test_gemini"),
                }
                exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), env)
                result = env["GeminiARSResult"](
                    standard_text="알겠습니다. 감사합니다.", ars_reply_jeju="예, 고맙수다.",
                )
                generate.return_value = SimpleNamespace(parsed=result, candidates=[])
                history = [env["ConversationTurn"](
                    jeju_text="반갑수다", standard_text="반갑습니다", ars_reply_jeju="반갑습니다.",
                )]
                with self.assertLogs("test_gemini", level="INFO") as logs:
                    actual, _ = env["call_gemini_ars"]("알았수다 고맙수다", history)
                self.assertIs(actual, result)
                request = generate.call_args.kwargs
                self.assertEqual(serialize(request["contents"])[:6], SOCIAL_TURNS)
                self.assertEqual(request["contents"][-1].parts[0].text, "민원인 제주어 질문: 알았수다 고맙수다")
                self.assertEqual(request["config"].response_mime_type, "application/json")
                message = logs.records[0].getMessage()
                if refs:
                    self.assertIn("동적 Few-Shot 4쌍 (공통 인사 예시 3쌍)", message)
                else:
                    self.assertIn("기존 Few-Shot (RAG 예시 없음)", message)


if __name__ == "__main__":
    unittest.main()
