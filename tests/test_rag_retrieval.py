"""Exercise retrieval without api_server's import-time GPU/model startup."""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "api_server.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "retrieve_context"
        )
        # Execute the real function body; only its external service is mocked.
        self.query = Mock()
        self.environment = {
            "RAG_CORPUS": "projects/test/locations/test/ragCorpora/test",
            "RAG_TOP_K": 3,
            "RAG_DISTANCE_THRESHOLD": 0.25,
            "time": SimpleNamespace(perf_counter=lambda: 0),
            "logger": logging.getLogger("test_retrieval"),
            "rag": SimpleNamespace(
                retrieval_query=self.query,
                RagResource=lambda **kwargs: kwargs,
                RagRetrievalConfig=lambda **kwargs: kwargs,
                Filter=lambda **kwargs: kwargs,
            ),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), self.environment)
        self.retrieve = self.environment["retrieve_context"]

    def test_preserves_source_and_text_and_logs_metadata(self):
        contexts = [SimpleNamespace(
            source_display_name="교통_대중교통요금.md", source_uri="gs://example/교통_대중교통요금.md",
            text="retrieved text", score=0.185,
        )]
        self.query.return_value = SimpleNamespace(contexts=SimpleNamespace(contexts=contexts))
        with self.assertLogs("test_retrieval", level="INFO") as logs:
            self.assertEqual(self.retrieve("버스 요금?"), [("교통_대중교통요금.md", "retrieved text")])
        message = logs.records[0].getMessage()
        self.assertIn(contexts[0].source_uri, message)
        self.assertIn("'chars': 14", message)
        self.assertIn("0.185", message)
        self.assertNotIn("retrieved text", message)

    def test_disabled_or_blank_query_does_not_call_service(self):
        self.assertEqual(self.retrieve("  \n"), [])
        self.environment["RAG_CORPUS"] = ""
        self.assertEqual(self.retrieve("질문?"), [])
        self.query.assert_not_called()

    def test_search_failure_returns_empty_references(self):
        self.query.side_effect = RuntimeError("synthetic retrieval failure")
        with self.assertLogs("test_retrieval", level="ERROR"):
            self.assertEqual(self.retrieve("질문?"), [])

    def test_no_matches_returns_empty_references(self):
        self.query.return_value = SimpleNamespace(contexts=SimpleNamespace(contexts=[]))
        self.assertEqual(self.retrieve("질문?"), [])


if __name__ == "__main__":
    unittest.main()
