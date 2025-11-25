# Missing Models in MLX Swift Port

This document provides a comprehensive overview of models present in the Python mlx-lm library but missing from the Swift port (mlx-swift-lm).

## Overview

**Python mlx-lm**: 94 model implementations
**Swift mlx-swift-lm**: 36 LLM models + 9 VLM models
**Missing**: ~58 LLM models

---

## Models Present in Swift

### LLM Models (36)
- BaichuanM1
- BailingMoe
- Bitnet
- Cohere
- DeepseekV3
- Ernie4_5
- Exaone4
- FalconH1
- Gemma
- Gemma2
- Gemma3nText (Gemma 3n)
- Gemma3Text (Gemma 3)
- GLM4
- GPTOSS (GPT2)
- Granite
- GraniteMoeHybrid
- Internlm2
- LFM2
- LFM2MoE
- Lille130m
- Llama
- MiMo
- NanoChat
- Olmo2
- OlmoE
- OpenELM
- Phi
- Phi3
- PhiMoE
- Qwen2
- Qwen3
- Qwen3MoE
- SmolLM3
- SSM (Mamba-related)
- Starcoder2
- RoPEUtils (utility)

### VLM Models (9)
- FastVLM
- Idefics3
- Paligemma
- Qwen2VL
- Qwen25VL
- Qwen3VL
- QwenVL
- Gemma3 (vision variant)
- SmolVLM2

---

## Missing Models (Categorized)

### Critical Missing Models (High Priority)

These are widely-used, state-of-the-art models that should be prioritized:

#### 1. Mixtral
**File**: `mixtral.py`
**Type**: Mixture of Experts (MoE)
**Description**: Mistral AI's MoE model with 8x7B architecture. One of the most popular open-source MoE models.
**Priority**: ⭐⭐⭐⭐⭐
**Use Cases**: General purpose, high quality text generation

#### 2. Mistral3
**File**: `mistral3.py`
**Type**: Dense transformer
**Description**: Latest Mistral model with improved performance
**Priority**: ⭐⭐⭐⭐⭐
**Use Cases**: General purpose, instruction following

#### 3. Llama4 (and Llama4_Text)
**Files**: `llama4.py`, `llama4_text.py`
**Type**: Dense transformer with vision support
**Description**: Meta's latest Llama model family with multimodal capabilities
**Priority**: ⭐⭐⭐⭐⭐
**Use Cases**: General purpose, state-of-the-art performance
**Note**: Swift has Llama but not Llama4

#### 4. Qwen Base
**File**: `qwen.py`
**Type**: Dense transformer
**Description**: Base Qwen model (Swift has Qwen2, Qwen3, but not original Qwen)
**Priority**: ⭐⭐⭐⭐
**Use Cases**: Chinese and English text generation

#### 5. Qwen2_MoE
**File**: `qwen2_moe.py`
**Type**: Mixture of Experts
**Description**: MoE variant of Qwen2 (Swift has Qwen3MoE but not Qwen2MoE)
**Priority**: ⭐⭐⭐⭐
**Use Cases**: Efficient large-scale generation

#### 6. Qwen3_Next
**File**: `qwen3_next.py`
**Type**: Dense transformer
**Description**: Advanced Qwen3 variant with architectural improvements
**Priority**: ⭐⭐⭐⭐
**Use Cases**: Latest Qwen features

#### 7. Mamba & Mamba2
**Files**: `mamba.py`, `mamba2.py`
**Type**: State Space Models (SSM)
**Description**: Linear-time sequence models, alternative to transformers
**Priority**: ⭐⭐⭐⭐
**Use Cases**: Long context, efficient inference
**Note**: Swift has SSM utilities but not full Mamba models

#### 8. Jamba
**File**: `jamba.py`
**Type**: Hybrid SSM + Transformer
**Description**: AI21 Labs' hybrid architecture combining Mamba and attention
**Priority**: ⭐⭐⭐⭐
**Use Cases**: Long context with efficiency

#### 9. MiniCPM & MiniCPM3
**Files**: `minicpm.py`, `minicpm3.py`
**Type**: Compact models
**Description**: OpenBMB's efficient small-scale models
**Priority**: ⭐⭐⭐
**Use Cases**: Edge deployment, mobile inference

#### 10. StableLM
**File**: `stablelm.py`
**Type**: Dense transformer
**Description**: Stability AI's language model
**Priority**: ⭐⭐⭐
**Use Cases**: General purpose, open weights

---

### Advanced/Research Models

Models with specialized architectures or experimental features:

#### 11. Cohere2
**File**: `cohere2.py`
**Type**: Dense transformer
**Description**: Cohere's Command model (newer version)
**Note**: Swift has Cohere but not Cohere2

#### 12. DeepseekV2
**File**: `deepseek_v2.py`
**Type**: MoE with innovations
**Description**: DeepSeek's V2 architecture with MLA (Multi-head Latent Attention)
**Note**: Swift has DeepseekV3 but not V2

#### 13. InternLM3
**File**: `internlm3.py`
**Type**: Dense transformer
**Description**: Latest InternLM model
**Note**: Swift has Internlm2 but not Internlm3

#### 14. GLM4_MoE
**File**: `glm4_moe.py`
**Type**: Mixture of Experts
**Description**: MoE variant of GLM4
**Note**: Swift has GLM4 but not MoE version

#### 15. Ernie4_5_MoE
**File**: `ernie4_5_moe.py`
**Type**: Mixture of Experts
**Description**: MoE variant of Baidu's ERNIE
**Note**: Swift has Ernie4_5 but not MoE version

#### 16. RecurrentGemma
**File**: `recurrent_gemma.py`
**Type**: Recurrent architecture
**Description**: Google's Griffin-based recurrent Gemma variant
**Priority**: ⭐⭐⭐
**Use Cases**: Efficient long-context processing

#### 17. LongCat_Flash
**File**: `longcat_flash.py`
**Type**: Long-context specialist
**Description**: Optimized for very long contexts with flash attention
**Priority**: ⭐⭐⭐
**Use Cases**: Document processing, long conversations

#### 18. Nemotron, Nemotron_H, NemotronNAS
**Files**: `nemotron.py`, `nemotron_h.py`, `nemotron-nas.py`
**Type**: Dense transformers
**Description**: NVIDIA's Nemotron model family
**Priority**: ⭐⭐⭐
**Use Cases**: High-performance inference

#### 19. OLMo, OLMo3
**Files**: `olmo.py`, `olmo3.py`
**Type**: Dense transformers
**Description**: Allen AI's fully open language models
**Note**: Swift has Olmo2 and OlmoE but not OLMo base and OLMo3

---

### Vision-Language Models (VLM)

Missing VLM models that exist in Python but not in Swift's VLM library:

#### 20. Pixtral
**File**: `pixtral.py`
**Type**: Vision-Language Model
**Description**: Mistral's multimodal model
**Priority**: ⭐⭐⭐⭐
**Use Cases**: Image understanding, visual question answering

#### 21. Kimi_VL
**File**: `kimi_vl.py`
**Type**: Vision-Language Model
**Description**: Moonshot AI's vision-language model
**Priority**: ⭐⭐⭐
**Use Cases**: Multimodal understanding

#### 22. LFM2-VL
**File**: `lfm2-vl.py`
**Type**: Vision-Language Model
**Description**: VL variant of LFM2
**Priority**: ⭐⭐
**Note**: Swift has LFM2 but not VL version

#### 23. Qwen3_VL_MoE
**File**: `qwen3_vl_moe.py`
**Type**: Vision-Language MoE
**Description**: MoE variant of Qwen3-VL
**Priority**: ⭐⭐⭐
**Note**: Swift has Qwen3VL but not MoE version

---

### Specialized/Domain-Specific Models

#### 24. GPT_BigCode
**File**: `gpt_bigcode.py`
**Type**: Code generation
**Description**: Specialized for code (StarCoder family)
**Priority**: ⭐⭐⭐
**Use Cases**: Code completion, generation

#### 25. GPT_NeoX
**File**: `gpt_neox.py`
**Type**: Dense transformer
**Description**: EleutherAI's GPT-NeoX architecture
**Priority**: ⭐⭐
**Use Cases**: General purpose, research

#### 26. DoTs1
**File**: `dots1.py`
**Type**: Specialized architecture
**Description**: Model with unique tokenization/architecture
**Priority**: ⭐
**Use Cases**: Research

#### 27. Klear
**File**: `Klear.py`
**Type**: Specialized model
**Description**: Domain-specific model
**Priority**: ⭐

#### 28. MiniMax
**File**: `minimax.py`
**Type**: Dense transformer
**Description**: MiniMax's language model
**Priority**: ⭐⭐
**Use Cases**: Chinese text generation

#### 29. Hunyuan & Hunyuan_v1_dense
**Files**: `hunyuan.py`, `hunyuan_v1_dense.py`
**Type**: Dense transformers
**Description**: Tencent's Hunyuan models
**Priority**: ⭐⭐⭐
**Use Cases**: Chinese and multilingual generation

#### 30. Kimi_Linear
**File**: `kimi_linear.py`
**Type**: Linear attention variant
**Description**: Moonshot AI's efficient attention mechanism
**Priority**: ⭐⭐
**Use Cases**: Long context efficiency

#### 31. PhiXtral
**File**: `phixtral.py`
**Type**: MoE
**Description**: Phi-based MoE model
**Priority**: ⭐⭐
**Note**: Swift has Phi, Phi3, PhiMoE but not PhiXtral

#### 32. PlaMo & PlaMo2
**Files**: `plamo.py`, `plamo2.py`
**Type**: Dense transformers
**Description**: Preferred Networks' models
**Priority**: ⭐⭐
**Use Cases**: Japanese text generation

#### 33. Seed_OSS
**File**: `seed_oss.py`
**Type**: Specialized model
**Description**: Open-source seed model
**Priority**: ⭐

---

### Smaller/Experimental Models

#### 34. AFM7
**File**: `afm7.py`
**Type**: Small model
**Description**: Apple Foundation Model (experimental)
**Priority**: ⭐⭐

#### 35. Apertus
**File**: `apertus.py`
**Type**: Experimental
**Priority**: ⭐

#### 36. BailingMoeLinear
**File**: `bailing_moe_linear.py`
**Type**: Linear attention MoE
**Description**: Variant of BailingMoe with linear attention
**Note**: Swift has BailingMoe but not Linear variant

#### 37. GatedDelta
**File**: `gated_delta.py`
**Type**: Delta-based model
**Description**: Experimental gating mechanism
**Priority**: ⭐

#### 38. Helium
**File**: `helium.py`
**Type**: Specialized architecture
**Priority**: ⭐

#### 39. Exaone (Base)
**File**: `exaone.py`
**Type**: Dense transformer
**Description**: Base Exaone model
**Note**: Swift has Exaone4 but not base Exaone

---

### Utility Models

#### 40. Gemma3 (Base)
**File**: `gemma3.py`
**Type**: Latest Gemma
**Description**: Base Gemma3 (without n suffix)
**Note**: Swift has Gemma3Text and Gemma3nText but may be missing base

#### 41. GraniteMoe
**File**: `granitemoe.py`
**Type**: MoE
**Description**: IBM's Granite MoE
**Note**: Swift has GraniteMoeHybrid but not base GraniteMoe

---

## Complete Missing Models List

Here's the exhaustive list of all 58+ missing models:

1. AFM7
2. Apertus
3. BailingMoeLinear
4. Cohere2
5. DeepseekV2
6. DoTs1
7. Ernie4_5_MoE
8. Exaone (base)
9. GatedDelta
10. Gemma3 (base VL)
11. GraniteMoe
12. GPT_BigCode
13. GPT_NeoX
14. Helium
15. Hunyuan
16. Hunyuan_v1_dense
17. InternLM3
18. Jamba
19. Kimi_Linear
20. Kimi_VL
21. Klear
22. LFM2-VL
23. Llama4
24. Llama4_Text
25. LongCat_Flash
26. Mamba
27. Mamba2
28. MiniCPM
29. MiniCPM3
30. MiniMax
31. Mistral3
32. Mixtral
33. NemotronNAS
34. Nemotron
35. Nemotron_H
36. OLMo
37. OLMo3
38. PhiXtral
39. Pixtral
40. PlaMo
41. PlaMo2
42. Qwen (base)
43. Qwen2_MoE
44. Qwen3_Next
45. Qwen3_VL_MoE
46. RecurrentGemma
47. Seed_OSS
48. StableLM
49. GLM4_MoE

Plus utility files:
- `switch_layers.py` - Shared utility for MoE models
- `pipeline.py` - Pipeline utilities

---

## Implementation Priority Recommendations

### Tier 1 (Immediate Priority)
These models are widely used and have significant community adoption:

1. **Mixtral** - Most popular open MoE model
2. **Llama4** - Latest Meta flagship
3. **Mistral3** - Latest Mistral flagship
4. **Mamba/Mamba2** - Important alternative architecture
5. **Jamba** - Hybrid architecture gaining adoption

### Tier 2 (High Priority)
Important models with good adoption:

6. **Qwen2_MoE** - Completes Qwen MoE family
7. **Qwen3_Next** - Latest Qwen variant
8. **Pixtral** - Important VLM
9. **DeepseekV2** - Innovative MLA architecture
10. **RecurrentGemma** - Efficient Gemma variant

### Tier 3 (Medium Priority)
Useful models for specific use cases:

11. **MiniCPM/MiniCPM3** - Edge deployment
12. **StableLM** - Popular open model
13. **GPT_BigCode** - Code generation
14. **InternLM3** - Latest InternLM
15. **Nemotron family** - NVIDIA ecosystem

### Tier 4 (Lower Priority)
Specialized or less commonly used:

- Regional models (Hunyuan, PlaMo, MiniMax)
- Experimental architectures
- Research models
- Variants of existing models

---

## Architecture Categories

### By Architecture Type

**Dense Transformers** (40+):
- Llama4, Mistral3, Qwen variants, DeepSeek, etc.

**Mixture of Experts** (12):
- Mixtral, Qwen2_MoE, Qwen3_VL_MoE, Jamba (hybrid), PhiXtral, GraniteMoe, GLM4_MoE, Ernie4_5_MoE, LFM2MoE

**State Space Models** (3):
- Mamba, Mamba2, Jamba (hybrid)

**Recurrent Models** (2):
- RecurrentGemma, LongCat_Flash

**Vision-Language** (4):
- Pixtral, Kimi_VL, LFM2-VL, Qwen3_VL_MoE

**Linear Attention** (2):
- Kimi_Linear, BailingMoeLinear

---

## Model Compatibility Notes

### Cross-Platform Considerations

When porting models from Python to Swift:

1. **Attention Mechanisms**: Ensure Flash Attention equivalents work on Metal
2. **Quantization**: Python models often use different quantization schemes
3. **RoPE Variants**: Many models use custom RoPE implementations
4. **MoE Routing**: Routing strategies differ between implementations
5. **Tokenizers**: Some models require specific tokenizer configurations

### Known Challenges

- **Mamba/SSM**: Requires special state handling
- **MoE Models**: Need efficient expert routing on Metal
- **Long Context**: Require memory-efficient implementations
- **VLM**: Need vision encoder integration

---

## Resources for Implementation

### Reference Implementations

- **Python Source**: `/Users/anthony/files/projects/forked/mlx-lm/mlx_lm/models/`
- **HuggingFace**: Model configs and weights
- **Model Papers**: Architecture details

### Testing

When porting a model:
1. Compare output logits with Python implementation
2. Test with known prompts and expected outputs
3. Verify tokenizer compatibility
4. Benchmark performance vs Python version

---

## Contribution Guidelines

To add a missing model:

1. Start with Tier 1-2 models
2. Use existing Swift models as templates
3. Ensure compatibility with Python saved weights
4. Add comprehensive tests
5. Document any platform-specific considerations
6. Update this list when complete

---

**Last Updated**: 2025-11-24
**Python Version**: mlx-lm (94 models)
**Swift Version**: mlx-swift-lm (36 LLM + 9 VLM models)
