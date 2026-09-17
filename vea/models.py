"""Loading VLMs and turning samples into model inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from vea.config import GenerationConfig, ModelSpec, VeaConfig, resolve_model

__all__ = ["LoadedModel", "load_model", "build_inputs", "generate_answer"]


@dataclass
class LoadedModel:
    """A model, its processor, and the spec they came from."""

    spec: ModelSpec
    model: object
    processor: object

    @property
    def device(self):
        return self.model.device

    @property
    def n_layers(self) -> int:
        """Number of language-model layers, i.e. the attention stack depth."""
        config = self.model.config
        text_config = getattr(config, "text_config", config)
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(text_config, attr, None)
            if isinstance(value, int):
                return value
        raise AttributeError(f"cannot determine layer count for {self.spec.name}")


def load_model(
    name_or_id: str,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    eager_attention: bool = True,
    vea_config: VeaConfig | None = None,
) -> LoadedModel:
    """Load a VLM and its processor.

    Args:
        name_or_id: short name from :data:`vea.config.MODELS`, or any HF id.
        device: torch device string. Defaults to CUDA when available.
        dtype: model weight dtype.
        eager_attention: request the eager attention implementation. Required for
            attribution -- fused kernels never materialise the attention matrix,
            so ``output_attentions=True`` silently yields nothing. Set ``False``
            for evaluation-only models to save memory.
        vea_config: supplies ``max_pixels``, which caps the visual token count.

    Returns:
        The :class:`LoadedModel`.
    """
    spec = resolve_model(name_or_id)
    config = vea_config or VeaConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(
        spec.hf_id, use_fast=True, max_pixels=config.max_pixels
    )
    model_kwargs = {"torch_dtype": dtype}
    if eager_attention:
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForImageTextToText.from_pretrained(spec.hf_id, **model_kwargs)
    model = model.eval().to(device)
    return LoadedModel(spec=spec, model=model, processor=processor)


def build_inputs(processor, image: Image.Image, question: str, prompt_template: str):
    """Apply the chat template to one image-question pair.

    Args:
        processor: the model's processor.
        image: the image to show the model. For Vea this is the *augmented* image.
        question: the raw question text.
        prompt_template: a template from :data:`vea.config.PROMPTS`.

    Returns:
        The processor output, on CPU.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_template.format(question=question)},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


@torch.no_grad()
def generate_answer(
    model, processor, inputs, generation: GenerationConfig | None = None
) -> tuple[str, int]:
    """Greedily decode an answer.

    Args:
        model: the loaded model.
        processor: its processor.
        inputs: output of :func:`build_inputs`, already on the model's device.
        generation: decoding settings.

    Returns:
        ``(answer_text, n_generated_tokens)``.
    """
    generation = generation or GenerationConfig()
    # Seeded even though decoding is greedy, so that any nondeterminism
    # introduced by a model's own sampling defaults is pinned down.
    torch.manual_seed(generation.seed)

    outputs = model.generate(
        **inputs,
        max_new_tokens=generation.max_new_tokens,
        num_beams=generation.num_beams,
        do_sample=generation.do_sample,
    )
    generated = outputs[:, inputs["input_ids"].shape[1] :]
    answer = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return answer.strip(), int(generated.shape[1])
