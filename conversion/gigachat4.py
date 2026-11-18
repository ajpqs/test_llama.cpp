from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, split_gdn_qkvz
from .deepseek import DeepseekV32Model


@ModelBase.register("GigaChat4ForCausalLM")
@ModelBase.example("ai-sage/GigaChat4-708B-A41B", "ai-sage/GigaChat4-37B-A4B")
class GigaChat4Model(DeepseekV32Model):
    model_arch = gguf.MODEL_ARCH.GIGACHAT4
    # the output norm that the MTP block shares is a gated norm
    mtp_shared_tensors = DeepseekV32Model.mtp_shared_tensors + (
        "model.norm.gate_up_projection.weight", "model.norm.gate_down_projection.weight")

    def set_vocab(self):
        self._set_vocab_gpt2()
        # transformers 5 ignores clean_up_tokenization_spaces for this ByteLevel decoder
        self.gguf_writer.remove_key(gguf.Keys.Tokenizer.CLEAN_SPACES)
        self.gguf_writer.add_clean_spaces(False)

    def get_vocab_base_pre(self, tokenizer) -> str:
        # the tokenizer hashes as "gigachat", which llama.cpp splits with the GPT-2 rules
        pre = super().get_vocab_base_pre(tokenizer)
        if pre != "gigachat":
            raise ValueError(f"expected the GigaChat tokenizer, got pre-tokenizer {pre!r}")
        return "gigachat4"

    def _tensor_requires_f32(self, new_name: str, bid: int | None, n_dims: int) -> bool:
        # the low-rank projections of the gated norms are not quantized, as in llama-quantize
        return new_name.endswith(("norm_gate_up.weight", "norm_gate_down.weight")) or super()._tensor_requires_f32(new_name, bid, n_dims)

    def get_tokenizer(self):
        # the checkpoint names the transformers 5 class TokenizersBackend, which AutoTokenizer of transformers 4 does not know
        from transformers import PreTrainedTokenizerFast
        return PreTrainedTokenizerFast.from_pretrained(self.dir_model)

    def set_gguf_parameters(self):
        # keys missing from config.json take the defaults of the reference GigaChat4Config
        for key, value in {
            "linear_attn_o_norm_eps": 1e-6,
            "swiglu_limit": 10.0,
            "final_logit_softcapping": 30.0,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 2048,
            "q_lora_rank": 1536,
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 2.5,
            "norm_topk_prob": True,
            "first_k_dense_replace": 3,
        }.items():
            self.hparams.setdefault(key, value)
        if self.hparams.get("full_attention_layers") is None:
            self.hparams["full_attention_layers"] = list(range(3, self.hparams["num_hidden_layers"], 4))

        for key, value in {
            "use_mla_scaling_factor": True,
            "rope_interleave": True,
            "hidden_act": "silu",
            "norm_type": "GatedNorm",
            "layernorm_type": "pre_post",
            "layernorm_gating_weight": 2.0,
            "gated_attention": True,
            "linear_attention_type": "Qwen3NextGatedDeltaNet",
            "linear_gating_type": "gated_rmsnorm_sigmoid",
            "linear_sigmoid_gate_scale": 2.0,
            "attention_bias": False,
            "use_shared_expert_sigmoid": False,
            "linear_use_short_conv": True,
            "linear_use_legacy_qkvz_layout": False,
        }.items():
            if self.hparams.get(key, value) != value:
                raise ValueError(f"GigaChat4 requires {key}={value!r}")
        norm_eps = self.hparams["linear_attn_o_norm_eps"]
        if norm_eps is not None and norm_eps != self.hparams["rms_norm_eps"]:
            raise ValueError("GigaChat4 requires linear_attn_o_norm_eps to match rms_norm_eps")
        # options of the HF config that the graph does not implement, None means the default
        for key, value in {
            "num_krope_heads": 1,
            "moe_layer_freq": 1,
            "topk_method": "noaux_tc",
            "attention_distillation": False,
            "nextn_is_sparse": False,
            "nextn_is_linear": False,
            "branch_multiplier": 1.0,
            "linear_attn_output_mult": 1.0,
            "indexer_norm_type": "LlamaRMSNorm",
            "indexer_norm_eps": 1e-6,
            "indexer_rotate_activation_enabled": False,
        }.items():
            if self.hparams.get(key, value) not in (None, value):
                raise ValueError(f"GigaChat4 requires {key}={value!r}")
        if self.rope_parameters.get("rope_type") != "yarn":
            raise ValueError("GigaChat4 only supports YaRN RoPE scaling")
        rope_factor = self.rope_parameters.get("factor")
        if rope_factor is None or not 1 <= rope_factor < float("inf"):
            raise ValueError("GigaChat4 requires a RoPE scaling factor >= 1")
        if self.rope_parameters.get("mscale", 1.0) != 1.0:
            raise ValueError("GigaChat4 requires RoPE mscale=1")
        if self.rope_parameters.get("mscale_all_dim") != 1.0:
            raise ValueError("GigaChat4 requires RoPE mscale_all_dim=1")
        if self.hparams["n_group"] is None or self.hparams["topk_group"] is None:
            raise ValueError("GigaChat4 requires n_group and topk_group")
        if self.hparams["q_lora_rank"] is None:
            raise ValueError("GigaChat4 requires q_lora_rank")

        # drop head_dim (== qk_rope_head_dim) so the key/value lengths are not derived from it
        self.hparams.pop("head_dim", None)
        if self.hparams["routed_scaling_factor"] == 0:
            raise ValueError("GigaChat4 does not support routed_scaling_factor=0")
        if self.hparams.get("scoring_func") is None:
            self.hparams["scoring_func"] = "sigmoid"
        sparse = bool(self.hparams["index_topk"])
        if not sparse:
            self.hparams.update({"index_n_heads": 0, "index_head_dim": 0, "index_topk": 0})
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_expert_shared_feed_forward_length(hparams["moe_intermediate_size"] * hparams["n_shared_experts"])

        self.gguf_writer.add_ssm_conv_kernel(hparams["linear_conv_kernel_dim"])
        self.gguf_writer.add_ssm_state_size(hparams["linear_key_head_dim"])
        self.gguf_writer.add_ssm_group_count(hparams["linear_num_key_heads"])
        self.gguf_writer.add_ssm_time_step_rank(hparams["linear_num_value_heads"])
        self.gguf_writer.add_ssm_inner_size(hparams["linear_value_head_dim"] * hparams["linear_num_value_heads"])

        # main layers not listed in full_attention_layers are recurrent, the MTP layer is not
        full_attention_layers = set(hparams["full_attention_layers"])
        n_main_layers = hparams["num_hidden_layers"]
        self.gguf_writer.add_recurrent_layers(
            [il < n_main_layers and il not in full_attention_layers for il in range(self.block_count)])

        swiglu_limit = float(hparams["swiglu_limit"] or 0.0)
        self.gguf_writer.add_swiglu_clamp_exp([swiglu_limit] * self.block_count)
        self.gguf_writer.add_swiglu_clamp_shexp([swiglu_limit] * self.block_count)

        # main layers only, the MTP block of a sparse model always has an indexer
        dense_layers = hparams.get("dense_attention_layers") or []
        self.gguf_writer.add_indexer_types(
            [sparse and il in full_attention_layers and il not in dense_layers for il in range(n_main_layers)])
        self.gguf_writer.add_final_logit_softcapping(hparams.get("final_logit_softcapping") or 0.0)

    def prepare_metadata(self, vocab_only: bool):
        from_dir = self.fname_out.is_dir()
        norm_gate_name = "model.norm.gate_up_projection.weight"
        if not vocab_only and norm_gate_name not in self.model_tensors:
            raise ValueError(f"GigaChat4 requires {norm_gate_name}")
        if norm_gate_name in self.model_tensors:
            self.gguf_writer.add_norm_gate_rank(self.model_tensors[norm_gate_name]().shape[0])
        super().prepare_metadata(vocab_only=vocab_only)

        if not self.mtp_only or not from_dir:
            return

        output_type: str = self.ftype.name.partition("_")[2]
        fname_default: str = gguf.naming_convention(
            self.metadata.name, self.metadata.basename, self.metadata.finetune,
            self.metadata.version, size_label=None, output_type=output_type, model_type=None)
        self.fname_out = self.fname_out.parent / f"mtp-{fname_default}.gguf"

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch)
        elif name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"
        elif name.endswith(".conv1d.weight"):
            data_torch = data_torch.squeeze()

        if name.endswith(".in_proj_qkvz.weight"):
            qkv, z = split_gdn_qkvz(
                data_torch, self.hparams["hidden_size"], self.hparams["linear_key_head_dim"],
                self.hparams["linear_value_head_dim"], self.hparams["linear_num_key_heads"], self.hparams["linear_num_value_heads"])
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_QKV,  bid, ".weight"), qkv)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_GATE, bid, ".weight"), z)
            return

        if name.endswith(".in_proj_ba.weight"):
            # original order:  [b, a] * head_count
            # corrected order: [b * head_count, a * head_count]
            num_v_heads = self.hparams["linear_num_value_heads"]
            num_k_heads = self.hparams["linear_num_key_heads"]
            hidden_size = self.hparams["hidden_size"]
            num_v_per_k = num_v_heads // num_k_heads
            data_torch = data_torch.permute(1, 0).contiguous()
            data_torch = data_torch.view(hidden_size, num_k_heads, 2 * num_v_per_k)
            beta, alpha = torch.split(data_torch, [num_v_per_k, num_v_per_k], dim=-1)
            beta = beta.contiguous().view(hidden_size, num_v_heads).permute(1, 0).contiguous()
            alpha = alpha.contiguous().view(hidden_size, num_v_heads).permute(1, 0).contiguous()
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_BETA,  bid, ".weight"), beta)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_ALPHA, bid, ".weight"), alpha)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def _map_fp8_weight_names(self, name: str) -> tuple[str, ...]:
        if name.endswith(".in_proj_qkvz.weight"):
            bid = int(name.split(".")[2])
            return (
                self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_QKV, bid, ".weight"),
                self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_GATE, bid, ".weight"),
            )
        if name.endswith(".in_proj_ba.weight"):
            bid = int(name.split(".")[2])
            return (
                self.format_tensor_name(gguf.MODEL_TENSOR.SSM_BETA, bid, ".weight"),
                self.format_tensor_name(gguf.MODEL_TENSOR.SSM_ALPHA, bid, ".weight"),
            )
        return super()._map_fp8_weight_names(name)
