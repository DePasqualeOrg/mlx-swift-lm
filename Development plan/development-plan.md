# Comprehensive Comparison Report: Python vs Swift MLX Libraries

## Executive Summary

The Swift port (`mlx-swift-lm`) is designed for deploying models on consumer devices (iOS/macOS apps). This context shapes what functionality matters most—efficient on-device inference, voice interaction capabilities, and responsive user experiences take priority over server infrastructure, batch processing, and developer tooling.

The Swift port provides solid inference capabilities for popular model architectures. The most significant gap for consumer applications is audio functionality (TTS/STT), which would enable voice-based app experiences.

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

## 3. MLX Audio (Audio Models) - Critical Gap for Consumer Apps

The Swift port has **zero audio functionality**. For consumer device deployment, this is the **most significant missing capability**—voice interaction is a core expectation for modern mobile/desktop apps.

### Missing TTS (Text-to-Speech) — All 9 Model Architectures

| Model | Priority | Notes |
|-------|----------|-------|
| **Kokoro** | **Critical** | Fast, multilingual (EN/JP/ZH), high quality, 24kHz |
| **Bark** | **High** | Zero-shot, expressive, voice cloning |
| Sesame/CSM | Medium | Conversational speech model |
| Spark | Medium | Multilingual with speaker encoding |
| OuteTTS | Medium | LLM-based (Llama/Qwen backend) |
| IndexTTS | Low | Speaker indexing, BigVGAN vocoder |
| Dia | Low | Diffusion-based synthesis |
| Llama TTS | Low | LLM-based |

### Missing STT (Speech-to-Text) — All 4 Model Architectures

| Model | Priority | Notes |
|-------|----------|-------|
| **Whisper** | **Critical** | Universal ASR, 99+ languages, best accuracy, multiple sizes |
| **Parakeet** | **High** | NVIDIA Conformer, fast, good for real-time |
| Voxtral | Low | |
| Wav2Vec | Low | Self-supervised, feature extraction |

### Missing Audio Codecs — All 7 Implementations

| Codec | Priority | Notes |
|-------|----------|-------|
| **Encodec** | **High** | Meta's neural codec, required by Bark |
| **DAC** | **High** | Descript Audio Codec, required by OuteTTS |
| BigVGAN | Medium | Vocoder, used by IndexTTS |
| Vocos | Medium | Vocoder |
| Mimi | Low | Multi-codec with transformer |
| SNAC | Low | Scalable neural audio codec |
| S3 | Low | Sparse spectrogram synthesizer |

### Missing Audio Features — Complete List

| Feature | Priority | Notes |
|---------|----------|-------|
| **Streaming TTS** | **Critical** | Real-time voice responses |
| **Real-time transcription** | **Critical** | Voice input for apps |
| **Audio preprocessing (STFT, mel filterbanks)** | **Critical** | Foundation for all audio models |
| Voice activity detection (VAD) | High | Hands-free interaction |
| Audio resampling | High | Sample rate conversion |
| Audio normalization | Medium | Volume leveling |
| Voice cloning | Medium | Personalized TTS |
| Speech-to-speech pipeline | Medium | Full voice assistant pipeline |
| Audio player/streaming | Medium | Playback utilities |

**Assessment:** Audio is the biggest gap. Whisper (STT) and Kokoro (TTS) would enable voice-first app experiences that users expect from modern applications. The audio preprocessing utilities (STFT, mel filterbanks) are foundational and needed before any audio model can work.

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

### Audio (Critical Gap)

Audio is the most significant missing capability for consumer apps.

| Priority | Item | Notes |
|----------|------|-------|
| Critical | Whisper STT | Voice input, dictation, accessibility |
| Critical | Kokoro TTS | Voice output, read-aloud, accessibility |
| Critical | Audio preprocessing | STFT, mel filterbanks (foundation for all audio) |
| High | Encodec/DAC codecs | Required by TTS models |
| High | Streaming audio | Real-time voice interaction |
| High | Parakeet STT | Fast real-time transcription |
| Medium | Bark TTS | Expressive voice synthesis |
| Medium | Voice activity detection | Hands-free interaction |
| Low | Voice cloning | Personalized TTS voices |

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
| MLX Audio | 20+ models | 0 | 0% | ❌ Critical gap for consumer apps |
| Embeddings | Via server | Dedicated library (15+ models) | ✅ | ✅ Swift has better support |

---

## Conclusion

For consumer device deployment, the Swift port is **well-positioned for text-based LLM and VLM inference**. The library covers popular model architectures and has essential features like streaming generation, KV cache management, and quantized model support.

**The critical gap is audio functionality.** Voice interaction (STT + TTS) is a core expectation for modern mobile and desktop applications. Adding Whisper and Kokoro would unlock:
- Voice assistants
- Dictation/transcription
- Accessibility features (screen readers, voice control)
- Hands-free interaction
- Audio content creation

**Recommended development focus:**
1. **First:** Audio foundation (preprocessing utilities, Encodec/DAC codecs)
2. **Then:** Whisper STT + Kokoro TTS
3. **Next:** Speculative decoding + prompt cache persistence
4. **Finally:** Mobile-efficient models (MiniCPM, Florence2, Phi3-V, Mamba)
