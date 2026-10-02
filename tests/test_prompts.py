"""Prompt layout, labels, rendering, and token-to-label mapping."""

import json
import os
import unittest

from nex.primitives import Choice, Noul, Score
from nex.prompts import CHOICE_LABELS, SYSTEM_PROMPT, build_prompt, render, token_to_slot

STATE = {"ticket": {"subject": "Charged twice", "messages": [{"from": "customer", "text": "Refund me, bitte. Grüße"}]}}


def choice(n):
    return Choice("Pick one.", {f"option{i}": None for i in range(n)})


class RenderTest(unittest.TestCase):
    def test_string_passes_through(self):
        self.assertEqual(render("plain text\nwith lines"), "plain text\nwith lines")

    def test_object_and_array_become_indented_json(self):
        self.assertEqual(render({"a": [1, 2]}), json.dumps({"a": [1, 2]}, indent=2))
        self.assertEqual(render(["x", {"y": 1}]), '[\n  "x",\n  {\n    "y": 1\n  }\n]')

    def test_non_ascii_kept(self):
        self.assertIn("Grüße", render(STATE))
        self.assertNotIn("\\u00fc", render(STATE))


class LabelsTest(unittest.TestCase):
    def test_choice_labels_are_letters(self):
        _, labels = build_prompt("s", Choice("x", {"billing": None, "tech": None, "sales": None}))
        self.assertEqual(labels, ["A", "B", "C"])
        _, labels = build_prompt("s", choice(20))
        self.assertEqual(labels, list("ABCDEFGHJKLMNOPQRSTU"))
        self.assertEqual(CHOICE_LABELS, "ABCDEFGHJKLMNOPQRSTU")

    def test_choice_labels_skip_i(self):
        # A reply that starts with the pronoun "I" must not count as an option.
        self.assertEqual(len(CHOICE_LABELS), 20)
        self.assertEqual(len(set(CHOICE_LABELS)), 20)
        self.assertNotIn("I", CHOICE_LABELS)
        _, labels = build_prompt("s", choice(9))
        self.assertEqual(labels, list("ABCDEFGHJ"))

    def test_score_labels_count_from_one(self):
        _, labels = build_prompt("s", Score("x", ["lo", "hi"]))
        self.assertEqual(labels, ["1", "2"])
        _, labels = build_prompt("s", Score("x", [str(i) for i in range(9)]))
        self.assertEqual(labels, [str(i) for i in range(1, 10)])
        # Ten levels do not fit the digits 1 to 9, so they keep 0 to 9.
        _, labels = build_prompt("s", Score("x", [str(i) for i in range(10)]))
        self.assertEqual(labels, [str(i) for i in range(10)])

    def test_noul_labels(self):
        _, labels = build_prompt("s", Noul("x"))
        self.assertEqual(labels, ["Yes", "No"])

    def test_unsupported_question_type(self):
        with self.assertRaises(TypeError):
            build_prompt("s", {"type": "noul", "instructions": "x"})


class LayoutTest(unittest.TestCase):
    def questions(self):
        return [
            Choice("Which team should handle this?", {"billing": "Payments and refunds", "tech": None}),
            Score("How urgent is this?", ["not urgent", "somewhat urgent", "very urgent"]),
            Noul("Is the customer asking for a refund?"),
            Noul({"check": "angry", "scale": "strict"}, {"true": "clearly upset"}),
        ]

    def test_two_messages_system_then_user(self):
        messages, _ = build_prompt(STATE, Noul("x"))
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)

    def test_state_is_an_identical_prefix_across_questions(self):
        prompts = [build_prompt(STATE, q)[0] for q in self.questions()]
        prefix = f"STATE:\n{render(STATE)}\n\nQUESTION:\n"
        for messages in prompts:
            self.assertEqual(messages[0], prompts[0][0])
            self.assertTrue(messages[1]["content"].startswith(prefix))
        users = [m[1]["content"] for m in prompts]
        common = os.path.commonprefix(users)
        self.assertTrue(common.startswith(prefix))
        # The shared prefix ends where the questions start to differ.
        self.assertLessEqual(len(common), len(prefix) + 3)

    def test_state_comes_before_question_and_options(self):
        q = Choice("Which team should handle this?", {"billing": None, "tech": None})
        user = build_prompt(STATE, q)[0][1]["content"]
        self.assertLess(user.index("Charged twice"), user.index("Which team should handle this?"))
        self.assertLess(user.index("Which team should handle this?"), user.index("OPTIONS:"))
        self.assertTrue(user.rstrip().endswith("of the best option."))

    def test_string_and_array_states(self):
        user = build_prompt("I was charged twice!", Noul("x"))[0][1]["content"]
        self.assertTrue(user.startswith("STATE:\nI was charged twice!\n\nQUESTION:\nx\n\n"))
        user = build_prompt(["first", "second"], Noul("x"))[0][1]["content"]
        self.assertIn(json.dumps(["first", "second"], indent=2), user)

    def test_object_instructions_rendered_as_json(self):
        instructions = {"task": "triage", "rules": ["be strict"]}
        user = build_prompt("s", Noul(instructions))[0][1]["content"]
        self.assertIn("QUESTION:\n" + json.dumps(instructions, indent=2) + "\n\nOPTIONS:", user)


class OptionLinesTest(unittest.TestCase):
    def user(self, question):
        return build_prompt("s", question)[0][1]["content"]

    def test_choice_options(self):
        user = self.user(Choice("x", {"billing": "Payments and refunds", "tech": None, "sales": "New deals"}))
        self.assertIn("OPTIONS:\nA) billing: Payments and refunds\nB) tech\nC) sales: New deals\n\n", user)
        self.assertIn("Reply with only the letter (A, B, or C) of the best option.", user)

    def test_choice_reply_lists_every_letter(self):
        user = self.user(Choice("x", {"yes": None, "no": None}))
        self.assertIn("Reply with only the letter (A or B) of the best option.", user)
        user = self.user(choice(9))
        self.assertIn("(A, B, C, D, E, F, G, H, or J)", user)
        user = self.user(choice(20))
        self.assertIn("(" + ", ".join("ABCDEFGHJKLMNOPQRST") + ", or U)", user)
        self.assertIn("T) option18\nU) option19\n", user)
        self.assertNotIn("I)", user)
        self.assertNotIn("-", user.rsplit("\n", 1)[-1])

    def test_choice_none_description_has_no_colon(self):
        user = self.user(Choice("x", {"billing": None, "tech": None}))
        self.assertIn("A) billing\nB) tech\n", user)
        self.assertNotIn("None", user)

    def test_choice_object_description_on_its_own_lines(self):
        desc = {"handles": ["refunds", "invoices"]}
        user = self.user(Choice("x", {"billing": desc, "tech": None}))
        self.assertIn("A) billing:\n" + json.dumps(desc, indent=2) + "\nB) tech", user)

    def test_score_options(self):
        user = self.user(Score("How urgent?", ["not urgent", "somewhat", "very"]))
        self.assertIn("OPTIONS:\n1) not urgent\n2) somewhat\n3) very\n\n", user)
        self.assertIn("\n\nReply with only the number (1-3) of the level that fits best.", user)
        self.assertNotIn("numbered from 0", user)

    def test_ten_level_score_says_it_starts_at_zero(self):
        user = self.user(Score("x", [f"level {i}" for i in range(10)]))
        self.assertIn("OPTIONS:\n0) level 0\n1) level 1\n", user)
        self.assertIn("9) level 9\n\nLevels are numbered from 0. Reply with only the number (0-9)", user)

    def test_score_object_level(self):
        user = self.user(Score("x", [{"label": "low"}, "high"]))
        self.assertIn("1) " + json.dumps({"label": "low"}, indent=2) + "\n2) high", user)

    def test_noul_options(self):
        self.assertIn("OPTIONS:\nYes\nNo\n\nReply with only Yes or No.", self.user(Noul("x")))
        user = self.user(Noul("x", {"true": "about money", "false": "anything else"}))
        self.assertIn("OPTIONS:\nYes: about money\nNo: anything else\n\n", user)
        user = self.user(Noul("x", {"false": "anything else", "true": None}))
        self.assertIn("OPTIONS:\nYes\nNo: anything else\n\n", user)
        self.assertNotIn("None", user)


class TokenToSlotChoiceTest(unittest.TestCase):
    def setUp(self):
        self.q = choice(3)

    def slot(self, token, n=3):
        return token_to_slot(token, self.q, n)

    def test_spellings(self):
        for token in ["A", " A", "a", " a", "A)", " A)", "A.", "A:", "A\n", "\tA "]:
            self.assertEqual(self.slot(token), 0, token)
        self.assertEqual(self.slot("B"), 1)
        self.assertEqual(self.slot(" c"), 2)

    def test_out_of_range_letters(self):
        for token in ["D", "d", "T", "Z", "z"]:
            self.assertIsNone(self.slot(token), token)
        self.assertEqual(token_to_slot("U", choice(20), 20), 19)
        self.assertEqual(token_to_slot("T", choice(20), 20), 18)
        self.assertEqual(token_to_slot("J", choice(20), 20), 8)
        self.assertIsNone(token_to_slot("V", choice(20), 20))
        self.assertIsNone(token_to_slot("U", choice(19), 19))

    def test_pronoun_i_is_never_a_label(self):
        q = choice(20)
        for token in ["I", " I", "i", " i", "I.", "I:"]:
            self.assertIsNone(token_to_slot(token, q, 20), repr(token))

    def test_multi_character_and_other_tokens(self):
        for token in ["AB", "Ab", "billing", "1", "Yes", "(A", "-", "A,"]:
            self.assertIsNone(self.slot(token), token)

    def test_empty_and_whitespace(self):
        for token in ["", " ", "\n", "\t ", ")", ".", " ):"]:
            self.assertIsNone(self.slot(token), repr(token))

    def test_non_ascii_letters_are_not_labels(self):
        # "ﬆ" upper-cases to "ST" and "ı" to "I". Neither is a label spelling.
        q = choice(20)
        for token in ["ﬆ", "ﬅ", "ı", "Ａ", "À", "Α"]:
            self.assertIsNone(token_to_slot(token, q, 20), repr(token))


class TokenToSlotScoreTest(unittest.TestCase):
    def setUp(self):
        self.q = Score("x", ["a", "b", "c", "d", "e"])

    def slot(self, token, n=5):
        return token_to_slot(token, self.q, n)

    def test_digits_count_from_one(self):
        for i in range(5):
            self.assertEqual(self.slot(str(i + 1)), i)
            self.assertEqual(self.slot(f" {i + 1}"), i)
            self.assertEqual(self.slot(f"{i + 1})"), i)
            self.assertEqual(self.slot(f"{i + 1}."), i)

    def test_out_of_range_digits(self):
        for token in ["0", " 0", "6", "9", " 7"]:
            self.assertIsNone(self.slot(token), token)
        q9 = Score("x", [str(i) for i in range(9)])
        self.assertEqual(token_to_slot("9", q9, 9), 8)
        self.assertIsNone(token_to_slot("0", q9, 9))

    def test_ten_levels_count_from_zero(self):
        q10 = Score("x", [str(i) for i in range(10)])
        for i in range(10):
            self.assertEqual(token_to_slot(str(i), q10, 10), i)

    def test_unicode_digits_return_none(self):
        for token in ["₂", "²", "٢", "２", "②", "½"]:
            self.assertIsNone(self.slot(token), repr(token))

    def test_other_tokens(self):
        for token in ["10", "01", "-1", "1.5", "A", "two", "", " ", "\n"]:
            self.assertIsNone(self.slot(token), repr(token))


class TokenToSlotNoulTest(unittest.TestCase):
    def setUp(self):
        self.q = Noul("x")

    def slot(self, token):
        return token_to_slot(token, self.q, 2)

    def test_yes_spellings(self):
        for token in ["Yes", " yes", "YES", "yes", "Y", "y", "true", "True", " TRUE", "Yes.", "Yes)", "Yes:"]:
            self.assertEqual(self.slot(token), 0, token)

    def test_no_spellings(self):
        for token in ["No", " no", "NO", "n", "N", "false", " False", "No."]:
            self.assertEqual(self.slot(token), 1, token)

    def test_other_tokens(self):
        for token in ["Maybe", "Ye", "Nope", "A", "0", "1", "", " ", "Yes!", "¿Sí"]:
            self.assertIsNone(self.slot(token), repr(token))


if __name__ == "__main__":
    unittest.main()
