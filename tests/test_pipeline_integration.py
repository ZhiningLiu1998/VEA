"""End-to-end wiring test using a stub model and processor.

The unit tests each cover one stage. This module checks that the stages compose:
that the image span found in the prompt indexes the same tokens the patch grid
describes, that evidence labels and attention scores are aligned patch-for-patch,
and that the augmented image comes back at the resolution the model expects.

A stub stands in for a real VLM so this runs on CPU in milliseconds. It implements
exactly the surface the pipeline touches, following the Qwen2.5-VL adapter's
contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vea.config import PROMPTS, VeaConfig, resolve_method
from vea.data import VQASample
from vea.models import LoadedModel
from vea.pipeline import Attributor, attribution_row, evaluate_sample, layer_stats_rows

IMAGE_TOKEN_ID = 151655
PATCH_SIZE = 28
IMAGE_SIDE = 112          # 112 / 28 = 4, so a 4x4 = 16 patch grid
N_PATCHES = (IMAGE_SIDE // PATCH_SIZE) ** 2
N_LAYERS = 8
N_HEADS = 2


class StubInputs(dict):
    """Minimal stand-in for a ``BatchFeature``: unpackable and device-movable."""

    def to(self, *args, **kwargs):
        return self


class StubTokenizer:
    """Maps whitespace-separated words to stable ids, reserving the image token."""

    image_token = "<|image_pad|>"

    def __init__(self):
        self.vocab = {self.image_token: IMAGE_TOKEN_ID}

    def _id(self, token: str) -> int:
        return self.vocab.setdefault(token, len(self.vocab) + 1000)

    def __call__(self, text, add_special_tokens: bool = True):
        return {"input_ids": [self._id(w) for w in text.split()]}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, -1)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        reverse = {v: k for k, v in self.vocab.items()}
        return " ".join(reverse.get(int(i), "") for i in ids)

    def batch_decode(self, sequences, skip_special_tokens: bool = False):
        return [self.decode(seq, skip_special_tokens) for seq in sequences]


class StubProcessor:
    """Emulates a ``Qwen2_5_VLProcessor`` closely enough for the pipeline."""

    image_token = StubTokenizer.image_token

    def __init__(self):
        self.tokenizer = StubTokenizer()
        self.image_processor = SimpleNamespace(
            _processor_class="Qwen2_5_VLProcessor",
            patch_size=14,
            merge_size=2,
            min_pixels=4 * 28 * 28,
            max_pixels=(IMAGE_SIDE * IMAGE_SIDE),
            do_convert_rgb=True,
        )

    def apply_chat_template(self, messages, **kwargs):
        """Build ``[prefix] [image tokens] [question] [suffix]``."""
        question_text = messages[0]["content"][1]["text"]
        prefix = self.tokenizer("USER :")["input_ids"]
        image = [IMAGE_TOKEN_ID] * N_PATCHES
        question = self.tokenizer(question_text)["input_ids"]
        suffix = self.tokenizer("ASSISTANT :")["input_ids"]
        ids = prefix + image + question + suffix
        return StubInputs(input_ids=torch.tensor([ids]))

    def batch_decode(self, sequences, skip_special_tokens: bool = False):
        return self.tokenizer.batch_decode(sequences, skip_special_tokens)


class StubModel:
    """Returns attention that concentrates on one known patch in deep layers."""

    #: The patches the deep layers "look at": a 2x2 cluster at grid cells
    #: (1,1), (1,2), (2,1), (2,2) on a 4x4 grid. Real evidence is spatially
    #: coherent, and an isolated patch would be removed by the denoiser.
    HOT_PATCHES = (5, 6, 9, 10)

    def __init__(self, answer: str = "nokia"):
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(num_hidden_layers=N_LAYERS)
        self._answer = answer
        self._processor = None

    def __call__(self, input_ids=None, output_attentions=False, use_cache=True, **kwargs):
        n_tokens = input_ids.shape[1]
        attention = torch.full((N_LAYERS, 1, N_HEADS, n_tokens, n_tokens), 0.01)
        image_start = 2  # "USER :" is two tokens
        for layer in range(N_LAYERS // 2, N_LAYERS):
            for patch in self.HOT_PATCHES:
                attention[layer, :, :, -1, image_start + patch] = 5.0
        return SimpleNamespace(attentions=[attention[i] for i in range(N_LAYERS)])

    def generate(self, input_ids=None, **kwargs):
        answer_ids = [self._processor.tokenizer._id(w) for w in self._answer.split()]
        return torch.cat([input_ids, torch.tensor([answer_ids])], dim=1)


@pytest.fixture
def loaded() -> LoadedModel:
    processor = StubProcessor()
    model = StubModel()
    model._processor = processor
    return LoadedModel(
        spec=SimpleNamespace(name="stub", hf_id="stub/stub", supports_attribution=True),
        model=model,
        processor=processor,
    )


@pytest.fixture
def sample(tmp_path) -> VQASample:
    path = tmp_path / "img.jpg"
    Image.new("RGB", (IMAGE_SIDE, IMAGE_SIDE), (120, 120, 120)).save(path)
    # A box covering exactly the 2x2 cluster of grid cells (1,1)-(2,2).
    return VQASample(
        sample_id=0,
        question="what is the brand",
        answers=["nokia"],
        image_path=path,
        boxes=[[PATCH_SIZE, PATCH_SIZE, 3 * PATCH_SIZE, 3 * PATCH_SIZE]],
        dataset="stub",
    )


@pytest.fixture
def attributor(loaded) -> Attributor:
    # Profile the deep half, which is where the stub concentrates its attention.
    return Attributor(loaded, layers=list(range(N_LAYERS // 2, N_LAYERS)), config=VeaConfig())


class TestAttend:
    def test_layout_matches_the_patch_grid(self, attributor, sample):
        attribution = attributor.attend(sample, PROMPTS["qa"])
        assert attribution.layout.n_image_tokens == N_PATCHES
        assert attribution.layout.n_patches == N_PATCHES
        assert attribution.layout.grid_size == (4, 4)

    def test_question_span_is_found_and_follows_the_image(self, attributor, sample):
        layout = attributor.attend(sample, PROMPTS["qa"]).layout
        assert layout.question_span[0] >= layout.image_span[1]
        assert layout.question_span[1] <= layout.n_input_tokens

    def test_attention_has_one_row_per_layer(self, attributor, sample):
        attribution = attributor.attend(sample, PROMPTS["qa"])
        assert attribution.attention.shape == (
            N_LAYERS,
            attribution.layout.n_input_tokens,
        )

    def test_evidence_labels_mark_the_annotated_patch(self, attributor, sample):
        labels = attributor.attend(sample, PROMPTS["qa"]).labels
        assert labels.n_evidence_patches == len(StubModel.HOT_PATCHES)
        assert all(labels.patch_mask[p] for p in StubModel.HOT_PATCHES)

    def test_input_image_matches_the_grid_geometry(self, attributor, sample):
        attribution = attributor.attend(sample, PROMPTS["qa"])
        height, width = attribution.layout.image_hw
        assert attribution.input_image.size == (width, height)


class TestAlignment:
    """The property that makes attribution meaningful: token index == patch index."""

    def test_attention_peak_lands_on_the_annotated_patch(self, attributor, sample):
        attribution = attributor.attend(sample, PROMPTS["qa"])
        scores = attributor.patch_scores(
            attribution, attributor.profiled_layers, denoise=False
        )
        assert int(np.argmax(scores)) in StubModel.HOT_PATCHES

    def test_attribution_is_scored_as_perfect(self, attributor, sample):
        attribution = attributor.attend(sample, PROMPTS["qa"])
        scores = attributor.patch_scores(attribution, attributor.profiled_layers, denoise=False)
        row = attribution_row(sample, attribution, scores, "vea", "stub")
        # The hot cluster is exactly the annotated evidence.
        assert row["auroc"] == pytest.approx(1.0)
        assert row["n_evd_tokens"] == len(StubModel.HOT_PATCHES)

    def test_shallow_layers_do_not_localise(self, attributor, sample):
        attribution = attributor.attend(sample, PROMPTS["qa"])
        shallow = attributor.patch_scores(attribution, [0, 1], denoise=False)
        # Uniform attention in shallow layers -> no preference for the evidence.
        assert shallow.std() == pytest.approx(0.0, abs=1e-6)


class TestAugment:
    def test_augmented_image_keeps_the_input_resolution(self, attributor, sample):
        attribution = attributor.attend(sample, PROMPTS["qa"])
        augmented = attributor.augment(attribution)
        assert augmented.size == attribution.input_image.size
        assert augmented.mode == "RGB"

    def test_evidence_region_stays_brighter_than_the_rest(self, loaded, sample):
        # sigma is relative to the shorter side, so the default 0.5 would blur
        # across this deliberately tiny 112px test image and flatten the map. A
        # bandwidth proportionate to the image keeps the test about the mechanism.
        attributor = Attributor(
            loaded, layers=[6, 7], config=VeaConfig(sigma=0.05)
        )
        attribution = attributor.attend(sample, PROMPTS["qa"])
        augmented = np.asarray(attributor.augment(attribution)).astype(float)
        # The hot cluster spans grid cells (1,1)-(2,2) -> pixels [28:84, 28:84].
        evidence = augmented[PATCH_SIZE : 3 * PATCH_SIZE, PATCH_SIZE : 3 * PATCH_SIZE].mean()
        corner = augmented[0:PATCH_SIZE, 0:PATCH_SIZE].mean()
        assert evidence > corner

    def test_wide_smoothing_trades_localisation_for_smoothness(self, loaded, sample):
        # Documents a real sensitivity: because sigma scales with the shorter
        # side, a large value lifts the background and shrinks the contrast
        # between evidence and non-evidence pixels.
        def contrast(sigma: float) -> float:
            attributor = Attributor(loaded, layers=[6, 7], config=VeaConfig(sigma=sigma))
            evidence_map = attributor.evidence_map(attributor.attend(sample, PROMPTS["qa"]))
            hot = np.zeros(evidence_map.shape, dtype=bool)
            hot[PATCH_SIZE : 3 * PATCH_SIZE, PATCH_SIZE : 3 * PATCH_SIZE] = True
            return float(evidence_map[hot].mean() - evidence_map[~hot].mean())

        assert contrast(0.05) > contrast(0.5)

    def test_evidence_map_is_bounded(self, attributor, sample):
        attribution = attributor.attend(sample, PROMPTS["qa"])
        evidence_map = attributor.evidence_map(attribution)
        assert evidence_map.shape == attribution.layout.image_hw
        assert 0.0 <= evidence_map.min() and evidence_map.max() <= 1.0

    def test_ablation_flags_reach_the_output(self, attributor, sample):
        attribution = attributor.attend(sample, PROMPTS["qa"])
        full = attributor.evidence_map(attribution)
        no_smooth = attributor.evidence_map(attribution, smooth=False)
        assert not np.allclose(full, no_smooth)

    def test_alpha_one_leaves_the_image_untouched(self, loaded, sample):
        plain = Attributor(loaded, layers=[6, 7], config=VeaConfig(alpha=1.0))
        attribution = plain.attend(sample, PROMPTS["qa"])
        np.testing.assert_array_equal(
            np.asarray(plain.augment(attribution)), np.asarray(attribution.input_image)
        )


class TestEvaluateSample:
    def test_base_method_needs_no_attributor(self, loaded, sample):
        row = evaluate_sample(sample, resolve_method("base"), loaded, None, PROMPTS)
        assert row["method"] == "base"
        assert row["em"] == 1.0  # the stub always answers "nokia"

    def test_augmenting_method_without_an_attributor_is_rejected(self, loaded, sample):
        with pytest.raises(ValueError, match="needs an attributor"):
            evaluate_sample(sample, resolve_method("vea"), loaded, None, PROMPTS)

    def test_row_carries_the_expected_columns(self, loaded, attributor, sample):
        row = evaluate_sample(sample, resolve_method("vea"), loaded, attributor, PROMPTS)
        assert {
            "sample_id", "dataset", "model", "method", "image_path", "question",
            "true_answer", "model_answer", "n_ans_tokens", "em", "f1",
        } <= row.keys()

    def test_cached_attribution_is_reused(self, loaded, attributor, sample):
        attribution = attributor.attend(sample, PROMPTS["qa"])
        row = evaluate_sample(
            sample, resolve_method("vea"), loaded, attributor, PROMPTS,
            attribution=attribution,
        )
        assert row["method"] == "vea"

    def test_every_method_runs(self, loaded, attributor, sample):
        from vea.config import METHODS

        for name in METHODS:
            row = evaluate_sample(sample, resolve_method(name), loaded, attributor, PROMPTS)
            assert row["method"] == name


class TestLayerStatsRows:
    def test_one_row_per_layer(self, attributor, sample):
        attribution = attributor.attend(sample, PROMPTS["qa"])
        rows = layer_stats_rows(sample, attribution, "stub")
        assert len(rows) == N_LAYERS
        assert [r["layer"] for r in rows] == list(range(N_LAYERS))

    def test_deep_layers_favour_evidence_over_non_evidence(self, attributor, sample):
        # This is the paper's diagnostic, reproduced against a known ground truth.
        attribution = attributor.attend(sample, PROMPTS["qa"])
        rows = layer_stats_rows(sample, attribution, "stub")
        shallow = rows[0]
        deep = rows[N_LAYERS - 1]
        assert shallow["image_evd_mean_norm"] == pytest.approx(
            shallow["image_nonevd_mean_norm"], rel=1e-3
        )
        assert deep["image_evd_mean_norm"] > deep["image_nonevd_mean_norm"]

    def test_rows_carry_the_analysis_columns(self, attributor, sample):
        attribution = attributor.attend(sample, PROMPTS["qa"])
        row = layer_stats_rows(sample, attribution, "stub")[0]
        assert {
            "image_mean", "quest_mean", "image_evd_mean", "image_nonevd_mean",
            "image_mean_norm", "quest_mean_norm", "image_evd_mean_norm",
            "image_nonevd_mean_norm", "n_layers", "evd_token_ratio",
        } <= row.keys()


class TestProfiling:
    def test_profiling_recovers_the_grounded_layers(self, attributor, sample):
        from vea.profiling import profile_layers

        # The stub concentrates on the evidence patch only in the deep half, so
        # profiling must select from there.
        profile = profile_layers(
            attributor, [sample], config=VeaConfig(layer_top_fraction=0.5)
        )
        assert profile.n_layers == N_LAYERS
        assert all(layer >= N_LAYERS // 2 for layer in profile.layers)
        assert profile.mean_auroc_selected > profile.mean_auroc_all

    def test_samples_without_boxes_are_skipped(self, attributor, sample):
        from vea.profiling import profile_layers

        boxless = VQASample(
            sample_id=1, question=sample.question, answers=sample.answers,
            image_path=sample.image_path, boxes=[], dataset="stub",
        )
        with pytest.raises(ValueError, match="no usable diagnostic samples"):
            profile_layers(attributor, [boxless])

    def test_profile_round_trips_through_json(self, attributor, sample, tmp_path):
        from vea.profiling import load_profile, profile_layers, save_profile

        profile = profile_layers(attributor, [sample], config=VeaConfig(layer_top_fraction=0.5))
        path = save_profile(profile, tmp_path / "p.json")
        restored = load_profile(path)
        assert restored.layers == profile.layers
        assert restored.n_layers == profile.n_layers


class TestUnsupportedFamily:
    def test_unknown_processor_gives_an_actionable_error(self, loaded):
        from vea.adapters import get_adapter

        loaded.processor.image_processor._processor_class = "MysteryProcessor"
        with pytest.raises(NotImplementedError, match="no VisionAdapter"):
            get_adapter(loaded.processor)

    def test_grid_mismatch_is_caught(self, loaded, sample):
        from vea.layout import build_layout
        from vea.models import build_inputs

        image = sample.load_image()
        inputs = build_inputs(loaded.processor, image, sample.question, PROMPTS["qa"])
        # Break the geometry: a larger merge size implies fewer, bigger patches.
        loaded.processor.image_processor.merge_size = 4
        with pytest.raises(ValueError, match="does not match this processor"):
            build_layout(inputs, image, loaded.processor, sample.question)
