# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_files import GGUFModelFiles
from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import get_gguf_tensor_names
from .base import BaseGGUFWeightsAdapter, GGUFWeight

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

GEMMA4_TEXT_SUBSTR: dict[str, str] = {
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_q.": "self_attn.q_proj.",
    "attn_k.": "self_attn.k_proj.",
    "attn_v.": "self_attn.v_proj.",
    "attn_output.": "self_attn.o_proj.",
    "attn_norm.": "input_layernorm.",
    "post_attention_norm.": "post_attention_layernorm.",
    "ffn_norm.": "pre_feedforward_layernorm.",
    "post_ffw_norm_1.": "post_feedforward_layernorm_1.",
    "post_ffw_norm_2.": "post_feedforward_layernorm_2.",
    "post_ffw_norm.": "post_feedforward_layernorm.",
    "pre_ffw_norm_2.": "pre_feedforward_layernorm_2.",
    "ffn_gate_inp.scale": "router.scale",
    "ffn_gate_inp.": "router.proj.",
    "ffn_down_exps.scale": "router.per_expert_scale",
    "ffn_gate_up_exps.": "experts.0.gate_up_proj.",
    "ffn_down_exps.": "experts.0.down_proj.",
    "ffn_gate.": "mlp.gate_proj.",
    "ffn_up.": "mlp.up_proj.",
    "ffn_down.": "mlp.down_proj.",
    "layer_output_scale.weight": "layer_scalar",
    "layer_output_scale.": "layer_scalar.",
}

GEMMA4_VISION_SUBSTR: dict[str, str] = {
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_q.": "self_attn.q_proj.linear.",
    "attn_k.": "self_attn.k_proj.linear.",
    "attn_v.": "self_attn.v_proj.linear.",
    "attn_out.": "self_attn.o_proj.linear.",
    "ffn_gate.": "mlp.gate_proj.linear.",
    "ffn_up.": "mlp.up_proj.linear.",
    "ffn_down.": "mlp.down_proj.linear.",
    "ln1.": "input_layernorm.",
    "attn_post_norm.": "post_attention_layernorm.",
    "ln2.": "pre_feedforward_layernorm.",
    "ffn_post_norm.": "post_feedforward_layernorm.",
}


def build_gemma4_text_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_prefix={
            "token_embd.": "model.language_model.embed_tokens.",
            "blk.": "model.language_model.layers.",
            "output_norm.": "model.language_model.norm.",
            "output.": "lm_head.",
        },
        orig_to_new_substr=GEMMA4_TEXT_SUBSTR,
    )


def build_gemma4_vision_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_prefix={
            "v.position_embd.weight": (
                "model.vision_tower.patch_embedder.position_embedding_table"
            ),
            "v.std_bias": "model.vision_tower.std_bias",
            "v.std_scale": "model.vision_tower.std_scale",
            "v.patch_embd.": "model.vision_tower.patch_embedder.input_proj.",
            "v.blk.": "model.vision_tower.encoder.layers.",
            "mm.input_projection": "model.embed_vision.embedding_projection",
        },
        orig_to_new_substr=GEMMA4_VISION_SUBSTR,
    )


def build_gemma4_unified_text_mapper() -> WeightsMapper:
    """Text mapper for the encoder-free ``gemma4_unified`` variant.

    Same GGUF tensor names as the tower-based variant, but the checkpoint
    nests the language model under ``model.language_model.*`` and has no
    ``lm_head`` (weights are tied to the embedding), matching
    ``Gemma4UnifiedForConditionalGeneration.hf_to_vllm_mapper`` in vLLM.
    """
    return WeightsMapper(
        orig_to_new_prefix={
            "token_embd.": "model.language_model.embed_tokens.",
            "blk.": "model.language_model.layers.",
            "output_norm.": "model.language_model.norm.",
        },
        orig_to_new_substr=GEMMA4_TEXT_SUBSTR,
    )


def build_gemma4_unified_vision_mapper() -> WeightsMapper:
    """Vision mapper for ``gemma4_unified``.

    ``gemma4_unified`` is encoder-free: there is no ``vision_tower``, so the
    GGUF ``v.*`` tensors feed ``Gemma4UnifiedVisionEmbedder`` and ``mm.*``
    feed the two ``Gemma4MultimodalEmbedder`` instances.
    """
    return WeightsMapper(
        orig_to_new_prefix={
            "v.patch_embd.weight": "model.vision_embedder.patch_dense.weight",
            "v.patch_embd.bias": "model.vision_embedder.patch_dense.bias",
            "v.patch_norm.1.weight": "model.vision_embedder.patch_ln1.weight",
            "v.patch_norm.1.bias": "model.vision_embedder.patch_ln1.bias",
            "v.patch_norm.2.weight": "model.vision_embedder.patch_ln2.weight",
            "v.patch_norm.2.bias": "model.vision_embedder.patch_ln2.bias",
            "v.patch_norm.3.weight": "model.vision_embedder.pos_norm.weight",
            "v.patch_norm.3.bias": "model.vision_embedder.pos_norm.bias",
            "v.position_embd.weight": "model.vision_embedder.pos_embedding",
            "mm.input_projection": "model.embed_vision.embedding_projection",
            "mm.a.input_projection": "model.embed_audio.embedding_projection",
        },
    )


def _map_tensor_name(mapper: WeightsMapper, name: str) -> str | None:
    mapped = mapper.apply_list([name])[0]
    return mapped if mapped != name else None


class Gemma4GGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for Gemma 4 text and multimodal GGUF models.

    Handles both the tower-based ``gemma4`` and the encoder-free
    ``gemma4_unified`` variant. The two share GGUF tensor names but nest
    them differently in the HF checkpoint.
    """

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in ("gemma4", "gemma4_unified")

    @staticmethod
    def is_unified(config) -> bool:
        return config.model_type == "gemma4_unified"

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        return maybe_patch_hf_config_from_gguf(
            files.primary_backbone,
            hf_config,
            mmproj_path=files.mm_proj,
        )

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        unified = self.is_unified(model_config.hf_config)
        text_mapper = (
            build_gemma4_unified_text_mapper()
            if unified
            else build_gemma4_text_mapper()
        )
        vision_mapper = (
            build_gemma4_unified_vision_mapper()
            if unified
            else build_gemma4_vision_mapper()
        )
        name_map: dict[str, str] = {}
        unmapped: list[str] = []
        for name in sorted(get_gguf_tensor_names(files.all_files)):
            mapper = vision_mapper if name.startswith(("v.", "mm.")) else text_mapper
            if mapped := _map_tensor_name(mapper, name):
                name_map[name] = mapped
            else:
                unmapped.append(name)
        if unmapped:
            logger.warning(
                "No HF name for %d Gemma 4 GGUF tensor(s), skipping: %s",
                len(unmapped),
                unmapped,
            )
        return name_map

    @staticmethod
    def _split_expert_weights(
        name: str, weight: torch.Tensor
    ) -> Iterable[tuple[str, torch.Tensor]]:
        if ".experts.0.gate_up_proj." in name:
            gate_name = name.replace("gate_up_proj", "gate_proj")
            up_name = name.replace("gate_up_proj", "up_proj")
            if weight.ndim == 0:
                yield gate_name, weight
                yield up_name, weight
                return
            for expert_id, expert_weight in enumerate(weight.unbind()):
                gate_weight, up_weight = expert_weight.chunk(2, dim=0)
                expert = f".experts.{expert_id}."
                yield gate_name.replace(".experts.0.", expert), gate_weight
                yield up_name.replace(".experts.0.", expert), up_weight
            return

        if weight.ndim == 3 and ".experts.0." in name:
            for expert_id, expert_weight in enumerate(weight.unbind()):
                yield (
                    name.replace(".experts.0.", f".experts.{expert_id}."),
                    expert_weight,
                )
            return
        yield name, weight

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        # gemma4_unified's patch projection is a plain Linear built from raw pixel
        # patches, so the GGUF 3-D [out, h, w] tensor has to be flattened to 2-D.
        # The tower-based variant does the same for its input_proj.
        unified = self.is_unified(model_config.hf_config)
        # Both variants feed their vision patch projection a GGUF conv-shaped
        # tensor that flattens to 2-D; only the target parameter name differs.
        flatten_names = (
            {"model.vision_embedder.patch_dense.weight"}
            if unified
            else {"model.vision_tower.patch_embedder.input_proj.weight"}
        )
        for name, weight in weights:
            if name in flatten_names:
                weight = weight.flatten(1)
            if name == "model.vision_embedder.pos_embedding":
                # vLLM allocates [mm_posemb_size, 2, mm_embed_dim] = [1120, 2, 3840].
                # The tensor handed to transform_weights is [1120, 3840, 2], not the
                # raw GGUF header order [3840, 1120, 2]: the plugin's GGUF linear
                # path has already moved mm_embed_dim to the end. Only the two
                # remaining axes need swapping.
                weight = weight.permute(1, 0, 2).contiguous()
            yield from self._split_expert_weights(name, weight)
