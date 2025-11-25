# Comprehensive Comparison Report: Python vs Swift MLX Libraries

## Executive Summary

The Swift port (`mlx-swift-lm`) is designed for deploying models on consumer devices (iOS/macOS apps). This context shapes what functionality matters most—efficient on-device inference, voice interaction capabilities, and responsive user experiences take priority over server infrastructure, batch processing, and developer tooling.

The Swift port provides solid inference capabilities for popular model architectures. **Text-to-Speech (TTS) is now available** via the separate [mlx-audio](https://github.com/Blaizzy/mlx-audio) Swift library, which includes Kokoro, Orpheus, and streaming Marvis models. The remaining critical gap is **Speech-to-Text (STT)**—porting Whisper would complete the voice interaction story for consumer apps.

---

## 1. MLX LM (Language Models)

### Model Architectures

| Python mlx-lm | Swift mlx-swift-lm | Status |
|--------------|-------------------|--------|
| **~80 architectures** | **36 architectures** | ⚠️ Partial (~45%) |

#### Implemented in Swift (36 models)

| Model | Notes |
|-------|-------|
| BaichuanM1 | |
| BailingMoE | |
| Bitnet | 1-bit quantization |
| Cohere | |
| DeepseekV3 | |
| Ernie4_5 | |
| Exaone4 | |
| FalconH1 | |
| Gemma | |
| Gemma2 | |
| Gemma3Text | |
| Gemma3nText | Text-only (no vision/audio) |
| GLM4 | |
| GPTOSS | |
| Granite | |
| GraniteMoeHybrid | |
| InternLM2 | |
| LFM2 | |
| LFM2MoE | |
| Lille130m | Tiny model |
| Llama | Includes Mistral, Mixtral |
| MiMo | |
| NanoChat | Tiny model |
| Olmo2 | |
| OlmoE | |
| OpenELM | Apple's efficient model |
| Phi | |
| Phi3 | |
| PhiMoE | |
| Qwen2 | |
| Qwen3 | |
| Qwen3MoE | |
| SmolLM3 | Small/efficient |
| Starcoder2 | Code generation |

#### Missing from Swift — High Priority for Consumer Devices

These models are specifically designed for efficient mobile/edge deployment:

| Model | Priority | Notes |
|-------|----------|-------|
| **MiniCPM** | **Critical** | Explicitly designed for mobile deployment |
| **MiniCPM3** | **Critical** | Latest mobile-optimized version |
| **Mamba** | **High** | State-space model, efficient inference |
| **Mamba2** | **High** | Improved state-space architecture |
| **RecurrentGemma** | **High** | Hybrid RNN-Transformer, efficient for long context |
| **Phi3Small** | **High** | Smaller Phi variant |
| **StableLM** | **Medium** | Efficient small models available |

#### Missing from Swift — Complete List (41 models)

| Model | Priority | Notes |
|-------|----------|-------|
| AFM7 | Low | Specialized |
| Apertus | Low | Specialized |
| Cohere2 | Medium | Updated Cohere |
| DBRX | Low | Large MoE |
| Deepseek | Medium | Original Deepseek |
| DeepseekV2 | Medium | |
| Dots1 | Low | Specialized |
| Ernie4_5_MoE | Low | Large MoE variant |
| Exaone | Low | Original version |
| GatedDelta | Low | Experimental |
| Gemma3 (full) | Medium | Full Gemma3 vs text-only |
| Gemma3n (full) | Medium | Full multimodal |
| GLM | Low | Original GLM |
| GLM4_MoE | Low | Large MoE |
| GPT-BigCode | Low | Code model |
| GPT-NeoX | Low | Older architecture |
| GPT2 | Low | Legacy |
| GraniteMoE | Low | MoE variant |
| Helium | Low | Specialized |
| Hunyuan | Low | |
| HunyuanV1Dense | Low | |
| InternLM3 | Medium | Latest InternLM |
| Jamba | Medium | Hybrid Mamba-Transformer |
| KimiLinear | Low | Specialized |
| KimiVL | Low | Vision-language |
| Klear | Low | Specialized |
| LFM2-VL | Low | Vision variant |
| Llama4 | Medium | Latest Llama |
| Llama4Text | Medium | Text-only Llama4 |
| LongcatFlash | Low | Long context |
| MiniMax | Low | |
| Mistral3 | Medium | Latest Mistral |
| Mixtral | Medium | MoE (partially via Llama) |
| Nemotron | Low | NVIDIA model |
| NemotronH | Low | |
| NemotronNAS | Low | |
| Olmo | Low | Original OLMo |
| Olmo3 | Medium | Latest OLMo |
| Phixtral | Low | Phi MoE variant |
| Pixtral | Low | Vision model |
| Plamo | Low | Japanese model |
| Plamo2 | Low | |
| Qwen | Low | Original Qwen (v1) |
| Qwen2_MoE | Low | Large MoE variant |
| Qwen3_Next | Low | Experimental |
| SeedOSS | Low | Specialized |

**Assessment:** Good coverage of mainstream architectures (Llama, Gemma, Qwen, Phi). Key gaps are mobile-optimized models (MiniCPM) and efficient architectures (Mamba, RecurrentGemma).

### Generation Features

| Feature | Python | Swift | Consumer Priority | Notes |
|---------|--------|-------|-------------------|-------|
| Basic generation | ✅ | ✅ | Critical | |
| Stream generation | ✅ | ✅ | Critical | Essential for responsive UX |
| Temperature sampling | ✅ | ✅ | High | |
| Top-P (nucleus) | ✅ | ✅ | High | |
| Repetition penalty | ✅ | ✅ | High | |
| KV cache | ✅ | ✅ | Critical | |
| Rotating KV cache | ✅ | ✅ | Critical | Memory-bounded devices |
| KV cache quantization | ✅ | ✅ | Critical | Reduces memory footprint |
| Prefill step size | ✅ | ✅ | High | |
| Top-K sampling | ✅ | ❌ | Medium | Nice to have |
| Min-P sampling | ✅ | ❌ | Low | Advanced sampling |
| XTC sampling | ✅ | ❌ | Low | Advanced sampling |
| Logit bias | ✅ | ❌ | Medium | Useful for app-specific steering |
| Speculative decoding | ✅ | ❌ | **High** | Faster responses on-device |
| Prompt caching (save/load) | ✅ | ❌ | **High** | Session persistence across app launches |
| Batch generation | ✅ | ❌ | Low | Single-user on consumer devices |

### Training & Fine-tuning

| Feature | Python | Swift | Consumer Priority | Notes |
|---------|--------|-------|-------------------|-------|
| LoRA training | ✅ | ✅ | Medium | On-device personalization |
| LoRA fusion | ✅ | ✅ | Medium | |
| QLoRA (quant+LoRA) | ✅ | ✅ | Medium | Memory-efficient fine-tuning |
| DoRA training | ✅ | ❌ | Low | |
| Gradient checkpointing | ✅ | ❌ | Low | Memory optimization for training |
| Distributed training | ✅ | ❌ | N/A | Not applicable to consumer devices |
| Training callbacks | ✅ | ❌ | Low | |
| WandB/MLflow logging | ✅ | ❌ | N/A | Developer tooling |
| DWQ training | ✅ | ❌ | Low | |

**Assessment:** LoRA support enables on-device personalization, which is the relevant training use case for consumer apps.

### Quantization

| Feature | Python | Swift | Consumer Priority | Notes |
|---------|--------|-------|-------------------|-------|
| Load quantized models | ✅ | ✅ | Critical | Essential for device memory |
| Mixed quantization (load) | ✅ | ✅ | High | |
| AWQ quantization | ✅ | ❌ | N/A | Prepare models in Python |
| GPTQ quantization | ✅ | ❌ | N/A | Prepare models in Python |
| DWQ quantization | ✅ | ❌ | N/A | Prepare models in Python |
| Dynamic quantization | ✅ | ❌ | Low | |
| Model quantization (create) | ✅ | ❌ | N/A | Prepare models in Python |

**Assessment:** Loading quantized models is critical and supported. Creating quantized models can remain a Python-side developer workflow.

### Model Management

| Feature | Python | Swift | Consumer Priority | Notes |
|---------|--------|-------|-------------------|-------|
| HF Hub download | ✅ | ✅ | High | |
| Model conversion | ✅ | ❌ | N/A | Developer tooling |
| GGUF export | ✅ | ❌ | N/A | Developer tooling |
| Cache management | ✅ | ❌ | Medium | App storage management |
| Upload to Hub | ✅ | ❌ | N/A | Developer tooling |

### Server & API

| Feature | Python | Swift | Consumer Priority | Notes |
|---------|--------|-------|-------------------|-------|
| HTTP server | ✅ | ❌ | N/A | Not needed for on-device |
| OpenAI-compatible API | ✅ | ❌ | N/A | Not needed for on-device |
| Chat completions endpoint | ✅ | ❌ | N/A | Not needed for on-device |
| Embeddings endpoint | ✅ | ❌ | N/A | Not needed for on-device |

**Assessment:** Server features are not applicable to consumer device deployment.

### Evaluation

| Feature | Python | Swift | Consumer Priority | Notes |
|---------|--------|-------|-------------------|-------|
| Perplexity evaluation | ✅ | ⚠️ | Low | Basic loss eval present |
| LM-Eval integration | ✅ | ❌ | N/A | Developer tooling |
| Benchmarking | ✅ | ❌ | Low | Useful but not critical |

---

## 2. MLX VLM (Vision-Language Models)

### Model Architectures

| Python mlx-vlm | Swift mlx-swift-lm | Status |
|---------------|-------------------|--------|
| **32 architectures** | **9 architectures** | ⚠️ Partial (~28%) |

#### Implemented in Swift (9 models)

| Model | Notes |
|-------|-------|
| FastVLM | LLaVA-Qwen2 based |
| Gemma3 | Vision + text |
| Idefics3 | |
| PaliGemma | |
| Qwen2-VL | |
| Qwen2.5-VL | |
| Qwen3-VL | |
| QwenVL | Original |
| SmolVLM2 | Small/efficient |

#### Missing from Swift — High Priority for Consumer Devices

These are small/efficient VLMs ideal for mobile deployment:

| Model | Priority | Notes |
|-------|----------|-------|
| **Florence2** | **Critical** | Microsoft's efficient VLM, excellent for mobile |
| **Phi3-V** | **Critical** | Small, efficient, high quality |
| **SmolVLM** | **High** | Original small VLM |
| **LLaVA** | **High** | Most widely used VLM family |
| **Molmo** | **Medium** | Efficient architecture |

#### Missing from Swift — Complete List (23 models)

| Model | Priority | Notes |
|-------|----------|-------|
| Aya-Vision | Low | Multilingual |
| DeepSeek-VL-V2 | Medium | Good quality |
| DeepSeekOCR | Medium | OCR specialized |
| Gemma3n (VLM) | **High** | Full multimodal (vision+audio); Swift only has text-only variant |
| GLM4V | Low | |
| GLM4V-MoE | Low | Large MoE |
| Idefics2 | Low | Older version |
| InternVL-Chat | Medium | Strong performance |
| Kimi-VL | Low | |
| LFM2-VL | Low | |
| Llama4 (VL) | Medium | Meta's latest |
| LLaVA | **High** | Widely used |
| LLaVA-Bunny | Medium | Qwen2 variant |
| LLaVA-Next | Medium | Improved LLaVA |
| Mistral3 | Medium | |
| MLlama | Medium | Meta's Llama VLM |
| Molmo | **Medium** | Efficient |
| MultiModality | Low | SAM + Text |
| Phi3-V | **Critical** | Small, efficient |
| Pixtral | Medium | Mistral's VLM |
| Qwen3-VL-MoE | Low | Large MoE |

### VLM Features

| Feature | Python | Swift | Consumer Priority | Notes |
|---------|--------|-------|-------------------|-------|
| Image input | ✅ | ✅ | Critical | |
| Multi-image input | ✅ | ⚠️ | Medium | Some models only |
| Video understanding | ✅ | ✅ | High | |
| Audio input (Gemma-3n) | ✅ | ❌ | High | Multi-modal apps |
| Image preprocessing | ✅ | ✅ | Critical | |
| URL image loading | ✅ | ✅ | High | |
| Chat templates | ✅ | ✅ | High | |
| Stream generation | ✅ | ✅ | Critical | |
| VLM fine-tuning (LoRA) | ✅ | ❌ | Low | |

**Assessment:** Good VLM coverage with Qwen and Gemma families. Key gaps are Florence2 and Phi3-V (efficient mobile VLMs) and the widely-used LLaVA family.

---

## 3. MLX Audio (Audio Models) - Separate Swift Library Available

Audio functionality exists in a **separate Swift library**: [mlx-audio](https://github.com/Blaizzy/mlx-audio). This library provides TTS capabilities for iOS/macOS apps using MLX.

### Swift TTS Implementations (in mlx-audio)

| Model | Status | Notes |
|-------|--------|-------|
| **Kokoro** | ✅ Implemented | Full phoneme-based TTS, multiple voices, 24kHz |
| **Orpheus** | ✅ Implemented | Qwen2-based LLM + SNAC codec, 8 voices, emotional expressions |
| **Marvis** | ✅ Implemented | Streaming TTS with Mimi codec, low-latency |

### Swift Audio Codecs (in mlx-audio)

| Codec | Status | Notes |
|-------|--------|-------|
| **Mimi** | ✅ Implemented | Full streaming decoder, 24kHz, 8-32 codebooks |
| **SNAC** | ✅ Implemented | Complete implementation for Orpheus |

### Swift Audio Features (in mlx-audio)

| Feature | Status | Notes |
|---------|--------|-------|
| Streaming TTS | ✅ | Marvis with AsyncThrowingStream |
| Audio playback | ✅ | AVAudioEngine integration |
| Audio file saving | ✅ | Export capabilities |
| Multiple voices | ✅ | 8+ voice options across models |
| Emotional expressions | ✅ | Orpheus: laugh, sigh, cough, etc. |

### Still Missing in Swift

| Model/Feature | Priority | Notes |
|---------------|----------|-------|
| **Whisper STT** | **Critical** | No Swift STT implementation yet |
| **Parakeet STT** | **High** | Fast real-time transcription |
| Voice activity detection | High | Hands-free interaction |
| Bark TTS | Medium | Zero-shot, expressive (Python only) |
| CSM/Sesame TTS | Medium | Voice cloning (Python only) |

### Python-Only Implementations (in mlx-audio)

The Python side of mlx-audio has more extensive coverage:

**TTS Models (Python):**
- Kokoro, Bark, CSM/Sesame, Spark, OuteTTS, IndexTTS, Dia, Llama TTS

**STT Models (Python):**
- Whisper, Parakeet, Wav2Vec, Voxtral

**Audio Codecs (Python):**
- Mimi, SNAC, Encodec, DAC, BigVGAN, Vocos, S3

**Assessment:** The mlx-audio Swift library provides solid TTS capabilities with Kokoro, Orpheus, and streaming Marvis. The critical remaining gap is **Speech-to-Text (STT)**—porting Whisper to Swift would complete the voice interaction story for consumer apps.

---

## 4. Embeddings

| Python mlx-lm | Swift mlx-swift-lm | Status |
|--------------|-------------------|--------|
| Embeddings via server API | Dedicated MLXEmbedders library | ✅ Swift has more |

Python MLX LM provides embeddings through its server API using LLM models. Swift has a dedicated `MLXEmbedders` library with purpose-built embedding model support.

#### Model Architectures in Swift

| Model | Notes |
|-------|-------|
| BERT | Base transformer embeddings |
| NomicBert | Optimized BERT variant |
| Qwen3 | Embedding variant |

#### Pre-configured Models in Swift

| Model | Hub ID |
|-------|--------|
| BGE micro | TaylorAI/bge-micro-v2 |
| BGE small | BAAI/bge-small-en-v1.5 |
| BGE base | BAAI/bge-base-en-v1.5 |
| BGE large | BAAI/bge-large-en-v1.5 |
| BGE m3 | BAAI/bge-m3 |
| GTE tiny | TaylorAI/gte-tiny |
| MiniLM L6 | sentence-transformers/all-MiniLM-L6-v2 |
| MiniLM L12 | sentence-transformers/all-MiniLM-L12-v2 |
| Snowflake XS | Snowflake/snowflake-arctic-embed-xs |
| Snowflake LG | Snowflake/snowflake-arctic-embed-l |
| Nomic v1 | nomic-ai/nomic-embed-text-v1 |
| Nomic v1.5 | nomic-ai/nomic-embed-text-v1.5 |
| Multilingual E5 small | intfloat/multilingual-e5-small |
| Mixedbread large | mixedbread-ai/mxbai-embed-large-v1 |
| Qwen3 Embedding | mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ |

**Assessment:** Swift has excellent embedding support with a dedicated library, better suited for on-device semantic search, RAG applications, and similarity matching than Python's server-based approach.

---

## Priority Recommendations for Consumer Device Deployment

### Audio (Partially Addressed via mlx-audio)

TTS is now available via the separate mlx-audio Swift library. STT remains the critical gap.

| Priority | Item | Notes |
|----------|------|-------|
| Critical | **Whisper STT** | Voice input, dictation, accessibility — **not yet in Swift** |
| High | Parakeet STT | Fast real-time transcription — **not yet in Swift** |
| High | Voice activity detection | Hands-free interaction |
| ✅ Done | Kokoro TTS | Available in mlx-audio Swift |
| ✅ Done | Streaming TTS | Marvis in mlx-audio Swift |
| ✅ Done | Audio codecs (Mimi, SNAC) | Available in mlx-audio Swift |
| Medium | Bark TTS | Expressive voice synthesis (Python only) |
| Low | Voice cloning | Personalized TTS voices (CSM in Python only) |

### LLM Generation Features

| Priority | Item | Notes |
|----------|------|-------|
| High | Speculative decoding | Faster on-device generation |
| High | Prompt cache persistence | Session continuity across app launches |
| Medium | Top-K sampling | Common generation option |
| Medium | Logit bias | App-specific output steering |
| Low | Cache management | Storage cleanup utilities |

### LLM Model Coverage

| Priority | Item | Notes |
|----------|------|-------|
| High | MiniCPM/MiniCPM3 | Designed for mobile deployment |
| High | Mamba/Mamba2 | Efficient state-space models |
| Medium | RecurrentGemma | Efficient long-context |
| Low | Other missing architectures | See complete list above |

### VLM Model Coverage

| Priority | Item | Notes |
|----------|------|-------|
| High | Florence2 | Microsoft's efficient VLM, excellent for mobile |
| High | Phi3-V | Small, efficient, high quality |
| High | LLaVA family | Most widely used VLMs |
| Medium | Gemma3n (full VLM) | Multimodal with audio support |
| Low | Other missing architectures | See complete list above |

### Embeddings

| Priority | Item | Notes |
|----------|------|-------|
| Medium | Gemma Embedding | Google's 308M multilingual embedding model |

### Not Priorities for Consumer Deployment

These can remain Python-only developer tools:
- Server/API infrastructure
- Batch generation
- Model conversion and quantization tools
- Distributed training
- LM-Eval integration
- Hub upload
- Large MoE models (DBRX, GLM4-MoE, etc.)

---

## Summary Statistics

| Library | Python | Swift | Coverage | Consumer Assessment |
|---------|--------|-------|----------|---------------------|
| MLX LM | ~80 models | 36 models | ~45% | ✅ Good coverage of popular models |
| MLX VLM | 32 models | 9 models | ~28% | ⚠️ Adequate, needs efficient VLMs |
| MLX Audio TTS | 8 models | 3 models (mlx-audio) | ~38% | ✅ Kokoro, Orpheus, Marvis available |
| MLX Audio STT | 4 models | 0 | 0% | ❌ Critical gap — needs Whisper |
| Audio Codecs | 7 codecs | 2 codecs (mlx-audio) | ~29% | ✅ Mimi, SNAC available |
| Embeddings | Via server | Dedicated library (15+ models) | ✅ | ✅ Swift has better support |

---

## Conclusion

For consumer device deployment, the Swift ecosystem is **well-positioned for text-based LLM and VLM inference**, and now has **solid TTS capabilities** via the mlx-audio library. The mlx-swift-lm library covers popular model architectures with essential features like streaming generation, KV cache management, and quantized model support.

**TTS is now available** via the separate [mlx-audio](https://github.com/Blaizzy/mlx-audio) Swift library:
- Kokoro TTS (phoneme-based, multiple voices, 24kHz)
- Orpheus TTS (LLM-based, emotional expressions, 8 voices)
- Marvis TTS (streaming, low-latency with Mimi codec)
- Audio codecs: Mimi (streaming) and SNAC

**The remaining critical gap is Speech-to-Text (STT).** Porting Whisper to Swift would complete the voice interaction story and unlock:
- Voice assistants
- Dictation/transcription
- Accessibility features (voice control)
- Hands-free interaction

**Recommended development focus:**
1. **Critical:** Whisper STT Swift port (enables full voice interaction)
2. **High:** Speculative decoding + prompt cache persistence
3. **High:** Mobile-efficient models (MiniCPM, Florence2, Phi3-V)
4. **Medium:** Voice activity detection for hands-free interaction
