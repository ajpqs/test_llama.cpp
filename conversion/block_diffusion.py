from __future__ import annotations

from typing import Any, TypeVar

from .base import TextModel, gguf, logger

TextModelT = TypeVar("TextModelT", bound=type[TextModel])


class BlockDiffusionMixin:
    _block_diffusion_wrapped: bool = True
    _block_diffusion_overrides: dict[str, Any] = {}

    def set_gguf_parameters(self) -> None:
        super().set_gguf_parameters()
        self._write_block_diffusion_metadata()

    def _bd_hparam(self, *keys: str, default: Any = None) -> Any:
        overrides = getattr(type(self), "_block_diffusion_overrides", {}) or {}
        for key in keys:
            if key in overrides and overrides[key] is not None:
                return overrides[key]
        for key in keys:
            if (val := self.hparams.get(key)) is not None:
                return val
        return default

    def _write_block_diffusion_metadata(self) -> None:
        keys = gguf.Keys.BlockDiffusion

        if (mask_token_id := self._bd_hparam("block_diffusion_mask_token_id", "mask_token_id")) is not None:
            self.gguf_writer.add_uint32(keys.MASK_TOKEN_ID, int(mask_token_id))
            logger.info("block diffusion: mask_token_id=%s", mask_token_id)

        if (block_size := self._bd_hparam("block_diffusion_block_size", "diffusion_block_size")) is not None:
            self.gguf_writer.add_uint32(keys.BLOCK_SIZE, int(block_size))
            logger.info("block diffusion: block_size=%s", block_size)

        if (threshold := self._bd_hparam("block_diffusion_confidence_threshold")) is not None:
            self.gguf_writer.add_float32(keys.CONFIDENCE_THRESHOLD, float(threshold))
            logger.info("block diffusion: confidence_threshold=%s", threshold)


def wrap_model_class(
    base_cls: TextModelT,
    *,
    overrides: dict[str, Any] | None = None,
) -> TextModelT:
    if not issubclass(base_cls, TextModel):
        raise TypeError(f"{base_cls.__name__} must be a TextModel subclass")

    if getattr(base_cls, "_block_diffusion_wrapped", False):
        if overrides:
            merged = dict(getattr(base_cls, "_block_diffusion_overrides", {}))
            merged.update(overrides)
            base_cls._block_diffusion_overrides = merged
        return base_cls

    name = f"BlockDiffusion_{base_cls.__name__}"

    class BlockDiffusionModel(BlockDiffusionMixin, base_cls):
        _block_diffusion_wrapped = True
        _block_diffusion_overrides = dict(overrides or {})

    BlockDiffusionModel.__name__ = name
    BlockDiffusionModel.__qualname__ = name
    return BlockDiffusionModel
