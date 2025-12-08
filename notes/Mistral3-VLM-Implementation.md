# Mistral 3 VLM Implementation Plan

## Overview

Mistral 3 is a vision-language model (VLM) that combines:
- **Pixtral** vision encoder
- **Ministral 3** text/language model
- **Multi-modal projector** to bridge vision and text embeddings

The text-only component (`ministral3`) is implemented in `MLXLLM/Models/Mistral3.swift`. This document outlines the plan for implementing the full VLM in `MLXVLM`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Mistral 3 VLM                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐    ┌──────────────────────┐             │
│  │    Pixtral    │───▶│  Multi-Modal         │             │
│  │ Vision Model  │    │  Projector           │             │
│  └───────────────┘    │  ┌────────────────┐  │             │
│                       │  │ RMSNorm        │  │             │
│                       │  │ Patch Merger   │  │             │
│                       │  │ Linear + GELU  │  │             │
│                       │  │ Linear         │  │             │
│                       │  └────────────────┘  │             │
│                       └──────────┬───────────┘             │
│                                  │                          │
│                                  ▼                          │
│                       ┌──────────────────────┐             │
│                       │   Ministral 3        │             │
│                       │   Language Model     │             │
│                       │   (text-only)        │             │
│                       └──────────────────────┘             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components to Implement

### 1. Pixtral Vision Model

The vision encoder used by Mistral 3. May already exist or need to be implemented in MLXVLM.

**Reference:** `/Users/anthony/files/projects/forked/mlx-vlm/mlx_vlm/models/pixtral/`

**Key features:**
- Patch-based image encoding
- Position embeddings for variable image sizes
- Transformer architecture

### 2. Mistral3PatchMerger

Merges spatial patches using learned weights.

**Reference:** `mlx-vlm/mlx_vlm/models/mistral3/mistral3.py` lines 109-166

```python
class Mistral3PatchMerger(nn.Module):
    """Learned merging of spatial_merge_size ** 2 patches"""

    def __init__(self, config: ModelConfig):
        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.vision_config.patch_size
        self.merging_layer = nn.Linear(
            hidden_size * self.spatial_merge_size**2,
            hidden_size,
            bias=False
        )
```

**Requires:** `unfold` operation (im2col) - see lines 19-106 in mistral3.py

### 3. Mistral3MultiModalProjector

Projects vision features to text embedding space.

**Reference:** `mlx-vlm/mlx_vlm/models/mistral3/mistral3.py` lines 169-200

```python
class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        self.norm = nn.RMSNorm(config.vision_config.hidden_size)
        self.patch_merger = Mistral3PatchMerger(config)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias
        )
```

### 4. Full VLM Model

Combines all components.

**Reference:** `mlx-vlm/mlx_vlm/models/mistral3/mistral3.py` lines 203-333

**Key methods:**
- `get_input_embeddings()` - Processes images and text, merges embeddings
- `merge_input_ids_with_image_features()` - Inserts image tokens into text sequence
- `sanitize()` - Weight key transformations for different model formats

## Configuration

**VLM Config Structure:**
```json
{
  "model_type": "mistral3",
  "image_token_index": 10,
  "spatial_merge_size": 2,
  "multimodal_projector_bias": false,
  "vision_feature_layer": -1,
  "text_config": {
    "model_type": "ministral3",
    "hidden_size": 3072,
    "num_hidden_layers": 26,
    ...
  },
  "vision_config": {
    "model_type": "pixtral",
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "patch_size": 14,
    ...
  }
}
```

## File Structure

```
Libraries/MLXVLM/Models/
├── Mistral3.swift          # Full VLM model (to be created)
├── Pixtral.swift           # Vision model (check if exists, else create)
└── ...

Libraries/MLXLLM/Models/
├── Mistral3.swift          # Text-only model (exists)
└── ...
```

## Implementation Steps

1. **Check for existing Pixtral implementation** in MLXVLM
2. **Implement Pixtral vision model** if not present
3. **Implement `unfold` utility** for patch operations
4. **Implement `Mistral3PatchMerger`**
5. **Implement `Mistral3MultiModalProjector`**
6. **Implement full `Mistral3Model`** in MLXVLM
7. **Add VLM configuration parsing**
8. **Register in VLM factory**
9. **Test with sample images**

## Reference Files

- Python VLM: `/Users/anthony/files/projects/forked/mlx-vlm/mlx_vlm/models/mistral3/`
- Python text model: `/Users/anthony/files/projects/forked/mlx-lm/mlx_lm/models/ministral3.py`
- Swift text model: `Libraries/MLXLLM/Models/Mistral3Text.swift` (`Mistral3TextModel`, `Mistral3TextConfiguration`)

## Notes

- The text-only model handles VLM configs by extracting from `text_config`
- Weight sanitization handles `language_model.*` prefix from VLM weights
- Image token merging replaces `[IMG]` tokens with projected vision features
