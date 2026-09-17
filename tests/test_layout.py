"""Tests for locating the image and question spans in a tokenized prompt."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vea.layout import find_question_span


class StubTokenizer:
    """Whitespace tokenizer with a fixed vocabulary, standing in for a real one.

    Real tokenizers would need weights downloaded; the span search only depends on
    subsequence matching, which this reproduces faithfully -- including the
    leading-space behaviour that makes the search non-trivial.
    """

    def __init__(self, leading_space_shifts_ids: bool = False):
        self.leading_space_shifts_ids = leading_space_shifts_ids
        self.vocab: dict[str, int] = {}

    def _id(self, token: str) -> int:
        return self.vocab.setdefault(token, len(self.vocab) + 100)

    def encode_text(self, text: str) -> list[int]:
        return [self._id(word) for word in text.split()]

    def __call__(self, text: str, add_special_tokens: bool = True):
        if self.leading_space_shifts_ids and text.startswith(" "):
            # Mimic BPE: a leading space produces a different first sub-token.
            words = text.split()
            ids = [self._id("_" + words[0])] + [self._id(w) for w in words[1:]]
        else:
            ids = self.encode_text(text)
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        reverse = {v: k for k, v in self.vocab.items()}
        return " ".join(reverse.get(i, "") for i in ids)


def make_processor(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(tokenizer=StubTokenizer(**kwargs))


class TestFindQuestionSpan:
    def test_locates_a_question_embedded_in_a_template(self):
        processor = make_processor()
        tokenizer = processor.tokenizer
        question = "what is the brand"
        prompt = f"USER : <image> answer this . {question} ASSISTANT :"
        input_ids = torch.tensor(tokenizer.encode_text(prompt))

        span = find_question_span(input_ids, processor, question)
        assert span is not None
        start, end = span
        expected = tokenizer.encode_text(question)
        assert input_ids[start:end].tolist() == expected

    def test_span_excludes_the_generation_prompt(self):
        processor = make_processor()
        question = "what colour"
        prompt = f"USER : {question} ASSISTANT :"
        input_ids = torch.tensor(processor.tokenizer.encode_text(prompt))

        _, end = find_question_span(input_ids, processor, question)
        assert end < len(input_ids)  # "ASSISTANT :" is not part of the question

    def test_prefers_the_last_occurrence(self):
        # Some templates echo the question; attention should be attributed to the
        # copy the model actually answers from, which is the last one.
        processor = make_processor()
        question = "how many"
        prompt = f"{question} ... USER : {question} ASSISTANT :"
        input_ids = torch.tensor(processor.tokenizer.encode_text(prompt))

        start, _ = find_question_span(input_ids, processor, question)
        assert start > 2

    def test_falls_back_when_the_question_is_absent(self):
        processor = make_processor()
        input_ids = torch.tensor(processor.tokenizer.encode_text("USER : hello ASSISTANT :"))
        assert find_question_span(input_ids, processor, "unrelated question") is None

    def test_tries_the_leading_space_variant(self):
        # With a space-sensitive tokenizer, the bare encoding does not match but
        # the space-prefixed one does; the search must try both.
        processor = make_processor(leading_space_shifts_ids=True)
        tokenizer = processor.tokenizer
        question = "what brand"
        # Build the prompt so the question appears in its space-prefixed form.
        prefix_ids = tokenizer.encode_text("USER :")
        question_ids = tokenizer(" " + question, add_special_tokens=False)["input_ids"]
        suffix_ids = tokenizer.encode_text("ASSISTANT :")
        input_ids = torch.tensor(prefix_ids + question_ids + suffix_ids)

        span = find_question_span(input_ids, processor, question)
        assert span == (len(prefix_ids), len(prefix_ids) + len(question_ids))

    def test_single_token_question_is_found(self):
        processor = make_processor()
        input_ids = torch.tensor(processor.tokenizer.encode_text("USER : why ASSISTANT :"))
        span = find_question_span(input_ids, processor, "why")
        assert span is not None and span[1] - span[0] == 1


class TestPromptLayout:
    def test_derived_counts_are_consistent(self):
        from tests.test_attention import make_layout

        layout = make_layout(image_span=(2, 6), grid_size=(2, 2))
        assert layout.n_image_tokens == 4
        assert layout.n_patches == 4
        assert len(layout.patch_coords) == layout.n_patches

    def test_layout_is_immutable(self):
        import dataclasses

        from tests.test_attention import make_layout

        with pytest.raises(dataclasses.FrozenInstanceError):
            make_layout().image_span = (0, 1)
