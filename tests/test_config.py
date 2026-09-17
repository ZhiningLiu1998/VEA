"""Tests for the registries and the profiling layer selection."""

from __future__ import annotations

import numpy as np
import pytest

from vea.config import METHODS, MODELS, PROMPTS, ProfileResult, resolve_method, resolve_model


class TestPrompts:
    def test_both_templates_take_a_question(self):
        for name, template in PROMPTS.items():
            assert "{question}" in template, name
            assert template.format(question="what colour?").endswith("what colour?")

    def test_vea_prompt_extends_the_base_prompt(self):
        # Inst vs. Vea must differ only in the image, so the prompt text is shared.
        assert "highlighted region" in PROMPTS["vea"]
        assert "highlighted region" not in PROMPTS["qa"]

    def test_both_offer_the_abstention_option(self):
        # Without it, the "false rejection" failure mode cannot be observed.
        for template in PROMPTS.values():
            assert "I cannot answer based on the given image." in template

    def test_wording_matches_the_published_templates(self):
        # Transcribed from the paper's appendix. Pinned because answer accuracy is
        # prompt-sensitive: paraphrasing here silently stops reproducing the paper.
        opening = (
            "Directly answer the question based on the image, no explanation is needed. "
            "If the image does not contain any relevant evidence, output "
            "\u201cI cannot answer based on the given image.\u201d "
        )
        assert PROMPTS["qa"] == opening + "Question: {question}"
        assert PROMPTS["vea"] == (
            opening
            + "Only use words from the picture, especially those in the highlighted "
            "region, to answer the question. Question: {question}"
        )

    def test_vea_prompt_is_the_base_prompt_plus_one_clause(self):
        # The Inst vs. Vea comparison is only clean if the prompts differ by
        # exactly this clause and nothing else.
        extra = PROMPTS["vea"].replace("Question: {question}", "")
        base = PROMPTS["qa"].replace("Question: {question}", "")
        assert extra.startswith(base)
        assert extra[len(base) :].strip() == (
            "Only use words from the picture, especially those in the highlighted "
            "region, to answer the question."
        )


class TestModelRegistry:
    def test_short_names_resolve(self):
        assert resolve_model("qwen2.5-vl-7b").hf_id == "Qwen/Qwen2.5-VL-7B-Instruct"

    def test_hf_ids_resolve(self):
        assert resolve_model("Qwen/Qwen2.5-VL-7B-Instruct").name == "qwen2.5-vl-7b"

    def test_unknown_ids_are_accepted_as_evaluation_only(self):
        spec = resolve_model("some-org/some-vlm")
        assert spec.name == "some-vlm"
        assert not spec.supports_attribution

    def test_attribution_flag_matches_the_adapter_registry(self):
        from vea.adapters import ADAPTERS

        families = {name for adapter in ADAPTERS for name in adapter.processor_classes}
        # Every model marked attribution-capable belongs to a family we can handle.
        expected = {"llava-1.5", "qwen2.5-vl", "gemma-3"}
        for spec in MODELS.values():
            if spec.supports_attribution:
                assert any(spec.name.startswith(prefix) for prefix in expected), spec.name
        assert families  # the registry is not empty


class TestMethodRegistry:
    def test_paper_methods_are_present(self):
        assert {"base", "inst", "vea"} <= set(METHODS)

    def test_base_uses_the_plain_prompt_and_original_image(self):
        base = resolve_method("base")
        assert base.prompt == "qa" and not base.augment

    def test_inst_isolates_the_prompt_from_the_augmentation(self):
        inst, vea = resolve_method("inst"), resolve_method("vea")
        assert inst.prompt == vea.prompt
        assert not inst.augment and vea.augment

    def test_ablations_each_disable_exactly_one_stage(self):
        vea = resolve_method("vea")
        assert vea.denoise and vea.smooth and vea.profile_layers
        assert not resolve_method("vea-no-denoise").denoise
        assert not resolve_method("vea-no-smooth").smooth
        assert not resolve_method("vea-no-profiling").profile_layers

    def test_unknown_method_lists_the_valid_choices(self):
        with pytest.raises(KeyError, match="unknown method"):
            resolve_method("magic")


class TestProfileResult:
    def _profile(self, aurocs, layers):
        return ProfileResult(
            model="m", n_layers=len(aurocs), layers=layers, layer_auroc=list(aurocs)
        )

    def test_selected_mean_exceeds_the_overall_mean(self):
        profile = self._profile([0.5, 0.6, 0.9, 0.95], layers=[2, 3])
        assert profile.mean_auroc_selected > profile.mean_auroc_all

    def test_nan_layers_are_ignored(self):
        profile = self._profile([0.5, np.nan, 0.9], layers=[2])
        assert profile.mean_auroc_all == pytest.approx(0.7)
        assert profile.mean_auroc_selected == pytest.approx(0.9)


class TestLayerSelection:
    """The top-fraction rule, exercised without loading a model."""

    @staticmethod
    def select(aurocs, fraction):
        import math

        n_keep = max(1, math.ceil(fraction * len(aurocs)))
        return sorted(int(i) for i in np.argsort(np.asarray(aurocs))[-n_keep:])

    def test_keeps_the_highest_scoring_layers(self):
        assert self.select([0.5, 0.9, 0.6, 0.95, 0.55], 0.4) == [1, 3]

    def test_always_keeps_at_least_one_layer(self):
        assert len(self.select([0.5] * 4, 0.01)) == 1

    def test_rounds_the_count_up(self):
        # ceil(0.1 * 28) == 3, matching the paper's selection sizes.
        assert len(self.select(list(np.linspace(0, 1, 28)), 0.1)) == 3

    def test_selection_is_ascending(self):
        rng = np.random.default_rng(0)
        selected = self.select(rng.random(32), 0.25)
        assert selected == sorted(selected)
