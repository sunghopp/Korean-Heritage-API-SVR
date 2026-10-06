"""Offline RAG regression tests using the real google-genai Content types.

Fixtures are synthetic v3 documents, not the production corpus. The fallback
snapshot was captured from pre-RAG commit b15ac45. No model or cloud calls run.
"""
import json
from pathlib import Path
import re
import unittest

from ars_prompt import RESPONSE_LANGUAGE_POLICY, RAG_SYSTEM_INSTRUCTION, build_prompt, parse_reference


QUESTIONS = [
    ("버스 요금 얼마우꽈?", "버스 요금은 얼마인가요?"),
    ("환승은 어떵 허우꽈?", "환승은 어떻게 하나요?"),
    ("어린이 할인 이수과?", "어린이 할인이 있나요?"),
    ("카드 어디서 사우꽈?", "카드는 어디서 사나요?"),
]
ANSWERS = ["요금 안내 예시이우다.", "환승 안내 예시이우다.", "할인 안내 예시이우다.", "카드 안내 예시이우다."]
SOURCE = "교통_대중교통요금.md"
CATEGORY = "교통 > 대중교통요금"
EXPECTED = [(jq, sq, answer) for (jq, sq), answer in zip(QUESTIONS, ANSWERS)]

# Expected public prompt payload, independent of the production case list.
SOCIAL_TURNS = [
    {"role": "user", "text": "민원 분야: 일상 인사\n민원인 제주어 질문: 안녕하세요. 반갑습니다."},
    {"role": "model", "text": "표준어 번역: 안녕하세요. 반갑습니다.\n만덕콜센터 제주어 답변: 예, 반갑수다. 궁금한 거 편하게 말씀해줍서."},
    {"role": "user", "text": "민원 분야: 인사와 민원 의도 확인\n민원인 제주어 질문: 예 반갑수다. 이번에 이디로 이사 와신디 전입신고 하젠 햄수다."},
    {"role": "model", "text": "표준어 번역: 예 반갑습니다. 이번에 여기로 이사 왔는데 전입신고 하려고 합니다.\n만덕콜센터 제주어 답변: 예, 반갑수다. 전입신고 문의로 확인해드리쿠다."},
    {"role": "user", "text": "민원 분야: 감사와 상담 마무리\n민원인 제주어 질문: 알았수다 고맙수다"},
    {"role": "model", "text": "표준어 번역: 알겠습니다. 감사합니다.\n만덕콜센터 제주어 답변: 예, 고맙수다. 다음에도 궁금한 거 편하게 말씀해줍서."},
]


def document(questions=QUESTIONS, answers=ANSWERS):
    return "\n".join([
        f"# {CATEGORY}", "## 관련 질문 예시 (제주어 / 표준어)",
        *[f"- {jq} / {sq}" for jq, sq in questions],
        "## 안내 정보", *[f"- 표준어 전용 안내 {i}" for i in range(4)],
        "## 제주어 안내 문구", *[f"- {answer}" for answer in answers],
    ])


def formats(text):
    plain = re.sub(r"(?m)^#+\s*|^-\s*", "", text)
    return {
        "markdown": text, "plain": plain,
        "flattened": text.replace("\n", " "),
        "flattened_plain": plain.replace("\n", " "),
        "crlf": text.replace("\n", "\r\n"),
    }


def serialize(contents):
    return [{"role": item.role, "text": item.parts[0].text} for item in contents]


class ParseReferenceTests(unittest.TestCase):
    def test_five_supported_formats(self):
        for label, text in formats(document()).items():
            with self.subTest(format=label):
                self.assertEqual(parse_reference(text, SOURCE), (CATEGORY, EXPECTED))

    def test_question_body_slashes_are_preserved(self):
        questions = [
            ("버스/택시 요금 얼마우꽈?", "버스/택시 요금은 얼마인가요?"),
            ("버스 / 택시 환승 어떵 허우꽈?", "버스 / 택시 환승은 어떻게 하나요?"),
            *QUESTIONS[2:],
        ]
        expected = [(jq, sq, answer) for (jq, sq), answer in zip(questions, ANSWERS)]
        for label, text in formats(document(questions)).items():
            with self.subTest(format=label):
                self.assertEqual(parse_reference(text, SOURCE), (CATEGORY, expected))

    def test_no_spaces_around_pair_separator(self):
        text = document().replace("? / ", "?/")
        self.assertEqual(parse_reference(text, SOURCE), (CATEGORY, EXPECTED))

    def test_heading_whitespace_and_bullet_variants(self):
        text = document().replace("관련 질문 예시", "관련 질문\n예시")
        text = text.replace("안내 정보", "안내\t정보").replace("제주어 안내 문구", "제주어\n안내 문구")
        text = text.replace("- ", "• ")
        self.assertEqual(parse_reference(text, SOURCE), (CATEGORY, EXPECTED))

    def test_language_label_is_optional(self):
        text = document().replace(" (제주어 / 표준어)", "")
        self.assertEqual(parse_reference(text, SOURCE), (CATEGORY, EXPECTED))

    def test_filename_takes_precedence_over_title(self):
        category, pairs = parse_reference(document(), "gs://bucket/생활_다른분야.MD")
        self.assertEqual(category, "생활 > 다른분야")
        self.assertEqual(pairs, EXPECTED)

    def test_title_fallback_stops_before_first_section(self):
        for name in ("", "display-name", "_missing-major.md"):
            for label, text in formats(document()).items():
                with self.subTest(name=name, format=label):
                    self.assertEqual(parse_reference(text, name), (CATEGORY, EXPECTED))

    def test_absent_title_produces_empty_category(self):
        text = document().split("\n", 1)[1]
        self.assertEqual(parse_reference(text, "display-name"), ("", EXPECTED))

    def test_incomplete_or_misaligned_documents_are_rejected(self):
        text = document()
        cases = {
            "missing_middle_answer": document(answers=[ANSWERS[0], *ANSWERS[2:]]),
            "missing_middle_question": document(questions=[QUESTIONS[0], *QUESTIONS[2:]]),
            "equal_but_partial_counts": document(QUESTIONS[:3], ANSWERS[:3]),
            "too_many_pairs": document(QUESTIONS + [QUESTIONS[0]], ANSWERS + [ANSWERS[0]]),
            "truncated_answer": text[:-3],
            "missing_question_mark": text.replace("환승은 어떵 허우꽈?", "환승은 어떵 허우꽈"),
            "missing_answer_ending": text.replace(ANSWERS[1], "환승 안내 예시예요."),
            "internal_period": text.replace(ANSWERS[1], "환승. 안내 예시이우다."),
            "trailing_unparsed_text": text + "\n미완성 답변",
            "questions_only": text[:text.index("## 안내 정보")],
            "answers_only": text[text.index("## 제주어 안내 문구"):],
            "missing_section": text.replace("## 안내 정보", "## 다른 제목"),
            "out_of_order_sections": text.replace("## 안내 정보", "## 제주어 안내 문구", 1),
            "concatenated_documents": text + "\n" + text,
            "empty": "",
        }
        for label, value in cases.items():
            with self.subTest(case=label):
                self.assertEqual(parse_reference(value, SOURCE)[1], [])


class PromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = json.loads(
            (Path(__file__).parent / "fixtures/pre_rag_prompt.json").read_text(encoding="utf-8")
        )

    def test_fallback_preserves_legacy_prompt_after_social_additions(self):
        for history_size in (0, 1, 5):
            history = [
                {"jeju_text": f"이전 질문 {i}", "standard_text": f"번역 {i}", "ars_reply_jeju": f"답변 {i}"}
                for i in range(history_size)
            ]
            expected = SOCIAL_TURNS + self.snapshot["contents"][:-1]
            for turn in history:
                expected.extend([
                    {"role": "user", "text": f"민원인 제주어 질문: {turn['jeju_text']}"},
                    {"role": "model", "text": f"표준어 번역: {turn['standard_text']}\n만덕콜센터 제주어 답변: {turn['ars_reply_jeju']}"},
                ])
            expected.append(self.snapshot["contents"][-1])
            for refs in (None, [], ["broken"], [(SOURCE, document(answers=ANSWERS[:3]))]):
                with self.subTest(history_size=history_size, references=refs):
                    system, contents = build_prompt("이번 질문?", history, refs)
                    self.assertEqual(system, self.snapshot["system_instruction"] + "\n\n" + RESPONSE_LANGUAGE_POLICY)
                    self.assertEqual(serialize(contents), expected)

    def test_dynamic_examples_then_history_then_question(self):
        history = [{"jeju_text": "이전 질문", "standard_text": "이전 번역", "ars_reply_jeju": "이전 답변"}]
        system, contents = build_prompt("이번 질문?", history, [
            (SOURCE, document()), ("생활_낮은관련도.md", document()),
        ])
        turns = serialize(contents)
        self.assertEqual(system, RAG_SYSTEM_INSTRUCTION)
        self.assertEqual(len(turns), 25)
        self.assertEqual(turns[:6], SOCIAL_TURNS)
        for offset, category in ((6, "생활 > 낮은관련도"), (14, CATEGORY)):
            for i, (jq, sq, answer) in enumerate(EXPECTED):
                self.assertEqual(turns[offset + 2 * i], {
                    "role": "user", "text": f"민원 분야: {category}\n민원인 제주어 질문: {jq}",
                })
                self.assertEqual(turns[offset + 2 * i + 1], {
                    "role": "model", "text": f"표준어 번역: {sq}\n만덕콜센터 제주어 답변: {answer}",
                })
        self.assertEqual(turns[-3]["text"], "민원인 제주어 질문: 이전 질문")
        self.assertEqual(turns[-2]["text"], "표준어 번역: 이전 번역\n만덕콜센터 제주어 답변: 이전 답변")
        self.assertEqual(turns[-1], {"role": "user", "text": "민원인 제주어 질문: 이번 질문?"})
        self.assertEqual([turn["role"] for turn in turns], ["user", "model"] * 12 + ["user"])
        self.assertNotIn("표준어 전용 안내", str(turns))

    def test_invalid_reference_does_not_discard_valid_reference(self):
        expected = build_prompt("질문?", references=[(SOURCE, document())])
        actual = build_prompt("질문?", references=[
            ("생활_손상문서.md", document(answers=ANSWERS[:3])), (SOURCE, document()),
        ])
        self.assertEqual(actual, expected)

    def test_plain_text_references_remain_supported(self):
        self.assertEqual(
            build_prompt("질문?", references=[document()]),
            build_prompt("질문?", references=[(SOURCE, document())]),
        )

    def test_greetings_keep_question_and_history_in_both_prompt_paths(self):
        history = [{
            "jeju_text": "반갑수다", "standard_text": "반갑습니다",
            "ars_reply_jeju": "반갑습니다. 무엇을 도와드릴까요?",
        }]
        for question in (
            "안녕하세요. 반갑습니다.",
            "예 반갑수다. 이번에 이디로 이사 와신디 전입신고 하젠 햄수다.",
            "알았수다 고맙수다",
            "감사합니다. 인터넷으로도 신청할 수 있나요?",
        ):
            for refs in ([], [(SOURCE, document())]):
                with self.subTest(question=question, rag=bool(refs)):
                    system, contents = build_prompt(question, history, refs)
                    turns = serialize(contents)
                    self.assertEqual(turns[:6], SOCIAL_TURNS)
                    self.assertEqual(system.count(RESPONSE_LANGUAGE_POLICY), 1)
                    self.assertIn("이전 대화에 표준어 AI 답변이 있더라도", system)
                    self.assertIn("감사 뒤에 질문이 이어지면 상담 종료로 간주하지 마세요", system)
                    self.assertEqual(turns[-3]["text"], "민원인 제주어 질문: 반갑수다")
                    self.assertEqual(turns[-2]["text"], "표준어 번역: 반갑습니다\n만덕콜센터 제주어 답변: 반갑습니다. 무엇을 도와드릴까요?")
                    self.assertEqual(turns[-1], {"role": "user", "text": f"민원인 제주어 질문: {question}"})

    def test_social_examples_separate_standard_translation_from_jeju_reply(self):
        for refs in ([], [(SOURCE, document())]):
            _, contents = build_prompt("고맙수다", references=refs)
            for turn in serialize(contents)[:6]:
                if turn["role"] != "model":
                    continue
                translation, reply = turn["text"].split("\n만덕콜센터 제주어 답변: ")
                self.assertTrue(translation.startswith("표준어 번역: "))
                self.assertNotRegex(reply, "반갑습니다|감사합니다|알겠습니다|전화주십시오")
            self.assertIn("표준어 번역: 알겠습니다. 감사합니다.", contents[5].parts[0].text)

    def test_partial_parse_logs_reason_and_counts(self):
        text = document(answers=[ANSWERS[0], *ANSWERS[2:]])
        with self.assertLogs("ars_prompt", level="WARNING") as logs:
            build_prompt("질문?", references=[(SOURCE, text)])
        message = logs.records[0].getMessage()
        for expected in (SOURCE, "expected_four_aligned_pairs", "questions=4", "answers=3"):
            self.assertIn(expected, message)

    def test_missing_sections_log_bounded_head_and_tail(self):
        text = "청크 시작\n" + "가" * 1000 + "\n청크 끝"
        with self.assertLogs("ars_prompt", level="WARNING") as logs:
            build_prompt("질문?", references=[(SOURCE, text)])
        message = logs.records[0].getMessage()
        self.assertIn("missing_repeated_or_out_of_order_sections", message)
        self.assertIn(f"chars={len(text)}", message)
        self.assertIn("questions=None answers=None", message)
        self.assertIn(repr(text[:300]), message)
        self.assertIn(repr(text[-300:]), message)
        self.assertNotIn("가" * 301, message)

    def test_valid_reference_does_not_warn(self):
        with self.assertNoLogs("ars_prompt", level="WARNING"):
            build_prompt("질문?", references=[(SOURCE, document())])


if __name__ == "__main__":
    unittest.main()
