import unittest

from sdft.eval.tooluse_metrics import score_tooluse_predictions


class ToolUseMetricsTest(unittest.TestCase):
    def test_single_action_key_order_does_not_break_strict_match(self):
        predictions = [
            "Thought: x\nAction: getSpecificVerse\nAction Input: {\"chapter\": 3, \"book\": \"John\", \"verse\": 16}"
        ]
        references = [
            [
                {
                    "Action": "getSpecificVerse",
                    "Action_Input": '{"book": "John", "chapter": 3, "verse": 16}',
                }
            ]
        ]

        scores = score_tooluse_predictions(predictions, references)

        self.assertEqual(scores["strict_match"], [1.0])
        self.assertEqual(scores["parse_success"], [1.0])
        self.assertEqual(scores["action_name_match"], [1.0])

    def test_multi_action_is_order_sensitive_for_strict_match(self):
        prediction = (
            "Action: first\nAction Input: {\"x\": 1}\n"
            "Action: second\nAction Input: {\"y\": 2}"
        )
        reversed_prediction = (
            "Action: second\nAction Input: {\"y\": 2}\n"
            "Action: first\nAction Input: {\"x\": 1}"
        )
        reference = [
            {"Action": "first", "Action_Input": '{"x": 1}'},
            {"Action": "second", "Action_Input": '{"y": 2}'},
        ]

        scores_ok = score_tooluse_predictions([prediction], [reference])
        scores_bad = score_tooluse_predictions([reversed_prediction], [reference])

        self.assertEqual(scores_ok["strict_match"], [1.0])
        self.assertEqual(scores_bad["strict_match"], [0.0])
        self.assertEqual(scores_bad["action_name_match"], [0.0])

    def test_invalid_action_input_sets_parse_failure_without_crash(self):
        predictions = ["Action: ping\nAction Input: {broken"]
        references = [[{"Action": "ping", "Action_Input": '{"ok": 1}'}]]

        scores = score_tooluse_predictions(predictions, references)

        self.assertEqual(scores["parse_success"], [0.0])
        self.assertEqual(scores["strict_match"], [0.0])

    def test_multiline_and_code_fenced_json_parses(self):
        predictions = [
            "Action: search\nAction Input: ```json\n{\n  \"q\": \"axolotl\",\n  \"page\": 1\n}\n```"
        ]
        references = [[{"Action": "search", "Action_Input": '{"page": 1, "q": "axolotl"}'}]]

        scores = score_tooluse_predictions(predictions, references)

        self.assertEqual(scores["parse_success"], [1.0])
        self.assertEqual(scores["strict_match"], [1.0])


if __name__ == "__main__":
    unittest.main()
