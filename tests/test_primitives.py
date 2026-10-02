"""Question validation, wire-format parsing, and answer serialization."""

import dataclasses
import json
import unittest

from nex.primitives import (
    MAX_CHOICE_OPTIONS,
    MAX_DEPTH,
    MAX_QUESTIONS,
    MAX_SCORE_LEVELS,
    Choice,
    ChoiceAnswer,
    Diagnostics,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    ValidationError,
    check_depth,
    question_from_dict,
    questions_from_dict,
)


def options(n):
    return {f"option{i}": None for i in range(n)}


def nested(depth, leaf="x"):
    """``leaf`` inside ``depth`` levels of alternating arrays and objects."""
    value = leaf
    for i in range(depth):
        value = [value] if i % 2 else {"k": value}
    return value


class ValidationAssertions(unittest.TestCase):
    def assertInvalid(self, field, fn, *args, contains=None):
        with self.assertRaises(ValidationError) as ctx:
            fn(*args)
        self.assertEqual(ctx.exception.field, field)
        if contains is not None:
            self.assertIn(contains, ctx.exception.message)
        return ctx.exception


class ValidationErrorTest(unittest.TestCase):
    def test_fields(self):
        e = ValidationError("questions.dept.criteria", "needs at least 2 options")
        self.assertIsInstance(e, ValueError)
        self.assertEqual(e.field, "questions.dept.criteria")
        self.assertEqual(e.message, "needs at least 2 options")
        self.assertEqual(str(e), "questions.dept.criteria: needs at least 2 options")


class InstructionsTest(ValidationAssertions):
    def test_string_object_and_array_instructions(self):
        for instructions in ["Which team?", {"task": "route", "rules": ["a", "b"]}, ["Pick a team.", "Be strict."]]:
            Choice(instructions, {"a": None, "b": None}).validate()
            Score(instructions, ["low", "high"]).validate()
            Noul(instructions).validate()

    def test_empty_and_blank_strings_rejected(self):
        for text in ["", "   ", "\n\t"]:
            for q in [Choice(text, {"a": None, "b": None}), Score(text, ["low", "high"]), Noul(text)]:
                self.assertInvalid("question.instructions", q.validate, contains="must not be empty")

    def test_wrong_types_rejected(self):
        for value in [None, 5, 1.5, True]:
            for q in [Choice(value, {"a": None, "b": None}), Score(value, ["low", "high"]), Noul(value)]:
                self.assertInvalid("question.instructions", q.validate, contains="string, object, or array")

    def test_where_prefix(self):
        self.assertInvalid("questions.x.instructions", Noul("").validate, "questions.x")

    def test_empty_object_and_array_rejected(self):
        for value in [{}, []]:
            for q in [Choice(value, {"a": None, "b": None}), Score(value, ["low", "high"]), Noul(value)]:
                self.assertInvalid("question.instructions", q.validate, contains="must not be empty")

    def test_nesting_depth_limit(self):
        self.assertEqual(MAX_DEPTH, 32)
        Noul(nested(MAX_DEPTH)).validate()
        for q in [Choice(nested(33), {"a": None, "b": None}), Score(nested(33), ["low", "high"]), Noul(nested(33))]:
            self.assertInvalid("question.instructions", q.validate, contains="32 levels")
        self.assertInvalid(
            "questions.x.instructions",
            questions_from_dict,
            {"x": {"type": "noul", "instructions": nested(33)}},
        )


class CheckDepthTest(ValidationAssertions):
    def test_counts_objects_and_arrays(self):
        for value in ["x", 1, None, {}, [], nested(1), nested(MAX_DEPTH), [nested(MAX_DEPTH - 1)] * 3]:
            check_depth(value, "state")
        for value in [nested(MAX_DEPTH + 1), [nested(MAX_DEPTH)], {"a": "x", "b": nested(MAX_DEPTH)}]:
            self.assertInvalid("state", check_depth, value, "state", contains="32 levels")

    def test_very_deep_input_does_not_recurse(self):
        self.assertInvalid("state", check_depth, nested(100_000), "state")

    def test_tuples_count_like_arrays(self):
        value = "x"
        for _ in range(MAX_DEPTH + 1):
            value = (value,)
        self.assertInvalid("state", check_depth, value, "state")

    def test_cycle_is_too_deep(self):
        cycle = {}
        cycle["self"] = cycle
        self.assertInvalid("state", check_depth, cycle, "state")


class ChoiceTest(ValidationAssertions):
    def test_valid(self):
        Choice("Which team?", {"billing": "Payments and refunds", "tech": None}).validate()

    def test_descriptions_may_be_string_object_array_or_none(self):
        Choice("x", {"a": "text", "b": {"examples": ["x"]}, "c": ["one", "two"], "d": None}).validate()

    def test_criteria_must_be_object(self):
        for criteria in [["a", "b"], "a,b", None, 3]:
            self.assertInvalid("question.criteria", Choice("x", criteria).validate, contains="must be an object")

    def test_option_limits(self):
        self.assertInvalid("question.criteria", Choice("x", options(0)).validate, contains="at least 2")
        self.assertInvalid("question.criteria", Choice("x", options(1)).validate, contains="at least 2")
        Choice("x", options(2)).validate()
        Choice("x", options(MAX_CHOICE_OPTIONS)).validate()
        self.assertEqual(MAX_CHOICE_OPTIONS, 20)
        self.assertInvalid("question.criteria", Choice("x", options(21)).validate, contains="at most 20")

    def test_option_names_must_be_non_empty_strings(self):
        for bad in ["", "  "]:
            self.assertInvalid("question.criteria", Choice("x", {bad: None, "b": None}).validate, contains="non-empty")
        self.assertInvalid("question.criteria", Choice("x", {1: None, "b": None}).validate, contains="non-empty")

    def test_description_errors_point_at_the_option(self):
        self.assertInvalid("question.criteria.billing", Choice("x", {"billing": "", "tech": None}).validate)
        self.assertInvalid("question.criteria.tech", Choice("x", {"billing": None, "tech": 4}).validate)

    def test_empty_object_or_array_description_rejected(self):
        for value in [{}, []]:
            self.assertInvalid(
                "question.criteria.tech", Choice("x", {"billing": None, "tech": value}).validate, contains="must not be empty"
            )

    def test_deep_description_rejected(self):
        Choice("x", {"billing": nested(MAX_DEPTH), "tech": None}).validate()
        self.assertInvalid(
            "question.criteria.billing", Choice("x", {"billing": nested(33), "tech": None}).validate, contains="32 levels"
        )

    def test_to_dict(self):
        q = Choice("Which team?", {"billing": "Payments", "tech": None})
        self.assertEqual(
            q.to_dict(), {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "tech": None}}
        )

    def test_frozen(self):
        q = Choice("x", {"a": None, "b": None})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            q.instructions = "y"


class ScoreTest(ValidationAssertions):
    def test_valid(self):
        Score("How urgent?", ["not urgent", "somewhat", "very"]).validate()

    def test_levels_may_be_string_object_or_array(self):
        Score("x", ["low", {"label": "mid", "examples": ["a"]}, ["high", "very high"]]).validate()

    def test_criteria_must_be_array(self):
        for criteria in [{"0": "low", "1": "high"}, "low,high", None, ("low", "high")]:
            self.assertInvalid("question.criteria", Score("x", criteria).validate, contains="must be an array")

    def test_level_limits(self):
        self.assertInvalid("question.criteria", Score("x", []).validate, contains="between 2 and 10")
        self.assertInvalid("question.criteria", Score("x", ["only"]).validate, contains="between 2 and 10")
        Score("x", ["a", "b"]).validate()
        Score("x", [str(i) for i in range(MAX_SCORE_LEVELS)]).validate()
        self.assertEqual(MAX_SCORE_LEVELS, 10)
        self.assertInvalid("question.criteria", Score("x", [str(i) for i in range(11)]).validate)

    def test_level_errors_point_at_the_index(self):
        self.assertInvalid("question.criteria[1]", Score("x", ["low", "", "high"]).validate, contains="must not be empty")
        self.assertInvalid("question.criteria[0]", Score("x", [None, "high"]).validate)
        self.assertInvalid("question.criteria[2]", Score("x", ["a", "b", 7]).validate)

    def test_empty_object_or_array_level_rejected(self):
        self.assertInvalid("question.criteria[1]", Score("x", ["low", {}]).validate, contains="must not be empty")
        self.assertInvalid("question.criteria[0]", Score("x", [[], "high"]).validate, contains="must not be empty")

    def test_deep_level_rejected(self):
        Score("x", ["low", nested(MAX_DEPTH)]).validate()
        self.assertInvalid("question.criteria[1]", Score("x", ["low", nested(33)]).validate, contains="32 levels")

    def test_to_dict(self):
        q = Score("How urgent?", ["low", "high"])
        self.assertEqual(q.to_dict(), {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]})


class NoulTest(ValidationAssertions):
    def test_criteria_optional(self):
        Noul("Is this about billing?").validate()
        self.assertIsNone(Noul("x").criteria)

    def test_true_and_false_keys(self):
        Noul("x", {"true": "about money", "false": "anything else"}).validate()
        Noul("x", {"true": "about money"}).validate()
        Noul("x", {"false": "anything else"}).validate()
        Noul("x", {"true": None}).validate()
        Noul("x", {}).validate()
        Noul("x", {"true": {"examples": ["refund"]}, "false": ["a", "b"]}).validate()

    def test_unknown_keys_rejected(self):
        for criteria in [{"yes": "x"}, {"true": "x", "no": "y"}, {"True": "x"}]:
            e = self.assertInvalid("question.criteria", Noul("x", criteria).validate, contains="unknown keys")
            self.assertIn('"true" and "false"', e.message)

    def test_criteria_must_be_object(self):
        for criteria in [["true", "false"], "yes", 1]:
            self.assertInvalid("question.criteria", Noul("x", criteria).validate, contains='"true" and "false"')

    def test_description_errors_point_at_the_key(self):
        self.assertInvalid("question.criteria.true", Noul("x", {"true": ""}).validate)
        self.assertInvalid("question.criteria.false", Noul("x", {"true": "ok", "false": 3}).validate)

    def test_empty_object_or_array_description_rejected(self):
        self.assertInvalid("question.criteria.true", Noul("x", {"true": {}}).validate, contains="must not be empty")
        self.assertInvalid("question.criteria.false", Noul("x", {"false": []}).validate, contains="must not be empty")

    def test_deep_description_rejected(self):
        Noul("x", {"true": nested(MAX_DEPTH)}).validate()
        self.assertInvalid("question.criteria.false", Noul("x", {"false": nested(33)}).validate, contains="32 levels")

    def test_to_dict_omits_missing_criteria(self):
        self.assertEqual(Noul("x").to_dict(), {"type": "noul", "instructions": "x"})
        self.assertEqual(
            Noul("x", {"true": "t"}).to_dict(), {"type": "noul", "instructions": "x", "criteria": {"true": "t"}}
        )


class QuestionFromDictTest(ValidationAssertions):
    def test_parses_each_type(self):
        self.assertEqual(
            question_from_dict({"type": "choice", "instructions": "x", "criteria": {"a": None, "b": "B"}}),
            Choice("x", {"a": None, "b": "B"}),
        )
        self.assertEqual(
            question_from_dict({"type": "score", "instructions": "x", "criteria": ["lo", "hi"]}), Score("x", ["lo", "hi"])
        )
        self.assertEqual(question_from_dict({"type": "noul", "instructions": "x"}), Noul("x"))
        self.assertEqual(
            question_from_dict({"type": "noul", "instructions": "x", "criteria": {"true": "t"}}), Noul("x", {"true": "t"})
        )

    def test_round_trip(self):
        questions = [
            Choice({"task": "route"}, {"billing": "Payments", "tech": ["bugs", "outages"], "other": None}),
            Score(["Rate urgency.", "Be strict."], ["low", {"label": "high"}]),
            Noul("Is it billing?", {"true": "money", "false": None}),
            Noul("Is it billing?"),
        ]
        for q in questions:
            wire = json.loads(json.dumps(q.to_dict()))
            self.assertEqual(question_from_dict(wire), q)

    def test_accepts_question_objects_and_validates_them(self):
        q = Noul("x")
        self.assertIs(question_from_dict(q), q)
        self.assertInvalid("questions.q.criteria", question_from_dict, Choice("x", options(1)), "questions.q")

    def test_must_be_object(self):
        for data in ["noul", ["noul"], None, 3]:
            self.assertInvalid("question", question_from_dict, data, contains="must be an object")

    def test_type_required_and_known(self):
        self.assertInvalid("question.type", question_from_dict, {"instructions": "x"})
        self.assertInvalid("question.type", question_from_dict, {"type": "boolean", "instructions": "x"})
        self.assertInvalid("question.type", question_from_dict, {"type": "Choice", "instructions": "x", "criteria": {}})
        self.assertInvalid("question.type", question_from_dict, {"type": None, "instructions": "x"})

    def test_unhashable_type_is_a_validation_error(self):
        # JSON can carry an array or object where the type string belongs.
        self.assertInvalid("question.type", question_from_dict, {"type": ["choice"], "instructions": "x"})
        self.assertInvalid("question.type", question_from_dict, {"type": {"name": "noul"}, "instructions": "x"})

    def test_instructions_required(self):
        for kind in ["choice", "score", "noul"]:
            self.assertInvalid("question.instructions", question_from_dict, {"type": kind}, contains="required")

    def test_criteria_required_for_choice_and_score(self):
        self.assertInvalid("question.criteria", question_from_dict, {"type": "choice", "instructions": "x"})
        self.assertInvalid("question.criteria", question_from_dict, {"type": "score", "instructions": "x"})

    def test_null_criteria_for_choice_is_a_type_error(self):
        self.assertInvalid(
            "question.criteria", question_from_dict, {"type": "choice", "instructions": "x", "criteria": None}
        )

    def test_unknown_fields_rejected(self):
        e = self.assertInvalid(
            "question", question_from_dict, {"type": "noul", "instructions": "x", "temperature": 0, "hint": "y"}
        )
        self.assertIn("hint", e.message)
        self.assertIn("temperature", e.message)


class QuestionsFromDictTest(ValidationAssertions):
    def test_parses_map(self):
        parsed = questions_from_dict(
            {
                "dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "tech": None}},
                "urgent": Noul("Is it urgent?"),
            }
        )
        self.assertEqual(list(parsed), ["dept", "urgent"])
        self.assertIsInstance(parsed["dept"], Choice)
        self.assertIsInstance(parsed["urgent"], Noul)

    def test_must_be_non_empty_object(self):
        for data in [{}, [], None, [{"type": "noul", "instructions": "x"}]]:
            self.assertInvalid("questions", questions_from_dict, data)

    def test_question_count_limit(self):
        self.assertEqual(MAX_QUESTIONS, 256)
        many = {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(MAX_QUESTIONS)}
        self.assertEqual(len(questions_from_dict(many)), MAX_QUESTIONS)
        many["one_more"] = {"type": "noul", "instructions": "x"}
        self.assertInvalid("questions", questions_from_dict, many, contains="at most 256")

    def test_too_many_questions_rejected_before_parsing_them(self):
        # The count is checked first, so the error names the limit, not a bad question.
        many = {f"q{i}": "not a question" for i in range(MAX_QUESTIONS + 1)}
        self.assertInvalid("questions", questions_from_dict, many, contains="at most 256")

    def test_deep_criteria_paths(self):
        self.assertInvalid(
            "questions.dept.criteria.billing",
            questions_from_dict,
            {"dept": {"type": "choice", "instructions": "x", "criteria": {"billing": nested(33), "tech": None}}},
        )
        self.assertInvalid(
            "questions.level.criteria[0]",
            questions_from_dict,
            {"level": {"type": "score", "instructions": "x", "criteria": [nested(40), "hi"]}},
        )

    def test_field_paths_name_the_question(self):
        self.assertInvalid(
            "questions.dept.criteria",
            questions_from_dict,
            {"dept": {"type": "choice", "instructions": "x", "criteria": {"billing": None}}},
        )
        self.assertInvalid(
            "questions.dept.criteria.billing",
            questions_from_dict,
            {"dept": {"type": "choice", "instructions": "x", "criteria": {"billing": "", "tech": None}}},
        )
        self.assertInvalid(
            "questions.level.criteria[1]",
            questions_from_dict,
            {"level": {"type": "score", "instructions": "x", "criteria": ["lo", ""]}},
        )
        self.assertInvalid(
            "questions.ok.instructions", questions_from_dict, {"ok": {"type": "noul", "instructions": "  "}}
        )
        self.assertInvalid("questions.ok.type", questions_from_dict, {"ok": {"type": "bool", "instructions": "x"}})
        self.assertInvalid("questions.ok", questions_from_dict, {"ok": "Is it billing?"})


class AnswerTest(unittest.TestCase):
    def test_choice_answer_to_dict_rounds(self):
        a = ChoiceAnswer(choice="billing", probabilities={"billing": 0.666666, "tech": 0.333334}, confidence=0.333332)
        self.assertEqual(
            a.to_dict(),
            {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.6667, "tech": 0.3333}, "confidence": 0.3333},
        )

    def test_score_answer_to_dict_rounds(self):
        a = ScoreAnswer(
            score=1.234567, legend={"0": "low", "1": "mid", "2": "high"},
            probabilities={"0": 0.1, "1": 0.555555, "2": 0.344445}, confidence=0.123456,
        )
        self.assertEqual(
            a.to_dict(),
            {
                "type": "score",
                "score": 1.2346,
                "legend": {"0": "low", "1": "mid", "2": "high"},
                "probabilities": {"0": 0.1, "1": 0.5556, "2": 0.3444},
                "confidence": 0.1235,
            },
        )

    def test_noul_answer_to_dict(self):
        self.assertEqual(NoulAnswer(noul=0.987654).to_dict(), {"type": "noul", "noul": 0.9877})

    def test_type_attributes(self):
        self.assertEqual((ChoiceAnswer.type, ScoreAnswer.type, NoulAnswer.type), ("choice", "score", "noul"))
        self.assertEqual((Choice.type, Score.type, Noul.type), ("choice", "score", "noul"))


class SystemOneResponseTest(unittest.TestCase):
    def setUp(self):
        self.choice = ChoiceAnswer("billing", {"billing": 0.8, "tech": 0.2}, 0.6)
        self.score = ScoreAnswer(1.5, {"0": "lo", "1": "mid", "2": "hi"}, {"0": 0.1, "1": 0.3, "2": 0.6}, 0.5)
        self.noul = NoulAnswer(0.25)
        self.diag = Diagnostics(
            label_mass=0.987654, raw_probabilities=[0.123456, 0.876544], prompt_tokens=90, cached_tokens=48,
            latency_ms=12.3456,
        )
        self.response = SystemOneResponse(
            model="nex-0.1.0+fake",
            answers={"dept": self.choice, "urgency": self.score, "billing": self.noul, "angry": NoulAnswer(0.9)},
            usage={"input_tokens": 300, "output_tokens": 4},
            diagnostics={"billing": self.diag},
        )

    def test_typed_views(self):
        self.assertEqual(self.response.choices, {"dept": self.choice})
        self.assertEqual(self.response.scores, {"urgency": self.score})
        self.assertEqual(list(self.response.nouls), ["billing", "angry"])
        self.assertIs(self.response.nouls["billing"], self.noul)

    def test_to_dict_omits_diagnostics_by_default(self):
        d = self.response.to_dict()
        self.assertEqual(set(d), {"model", "answers", "usage"})
        self.assertEqual(d["model"], "nex-0.1.0+fake")
        self.assertEqual(d["usage"], {"input_tokens": 300, "output_tokens": 4})
        self.assertEqual(d["answers"]["billing"], {"type": "noul", "noul": 0.25})
        self.assertEqual(d["answers"]["dept"]["choice"], "billing")
        self.assertEqual(list(d["answers"]), ["dept", "urgency", "billing", "angry"])
        json.dumps(d)

    def test_to_dict_with_diagnostics_rounds(self):
        d = self.response.to_dict(include_diagnostics=True)
        self.assertEqual(
            d["diagnostics"],
            {
                "billing": {
                    "label_mass": 0.9877,
                    "raw_probabilities": [0.1235, 0.8765],
                    "prompt_tokens": 90,
                    "cached_tokens": 48,
                    "latency_ms": 12.3,
                }
            },
        )

    def test_empty_diagnostics_default(self):
        r = SystemOneResponse(model="m", answers={}, usage={})
        self.assertEqual(r.diagnostics, {})
        self.assertEqual(r.to_dict(include_diagnostics=True)["diagnostics"], {})


if __name__ == "__main__":
    unittest.main()
