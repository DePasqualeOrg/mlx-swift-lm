// Copyright © 2024 Apple Inc.

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/ministral3.py

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Llama4 Attention Scaling

/// Compute attention scale for Llama 4 style position-based scaling.
///
/// - Parameters:
///   - start: Start position offset
///   - stop: Stop position offset
///   - beta: Scaling factor (llama_4_scaling_beta)
///   - maxPositionEmbeddings: Original max position embeddings
/// - Returns: Scaling tensor of shape [stop - start, 1]
private func getLlama4AttentionScale(
    start: Int, stop: Int, beta: Float, maxPositionEmbeddings: Int
) -> MLXArray {
    let positions = MLXArray(Int32(start) ..< Int32(stop))
    let scaling = 1 + beta * MLX.log(
        1 + MLX.floor(positions.asType(.float32) / Float(maxPositionEmbeddings))
    )
    return scaling[0..., .newAxis]
}

// MARK: - RoPE Initialization

/// Initialize RoPE based on rope_parameters configuration.
///
/// This function mirrors the Python `initialize_rope` function from rope_utils.py
private func initializeRope(
    dims: Int,
    base: Float,
    traditional: Bool,
    scalingConfig: [String: StringOrNumber]?,
    maxPositionEmbeddings: Int?
) -> Module {
    let ropeType: String
    if let config = scalingConfig,
        let typeValue = config["type"] ?? config["rope_type"],
        case .string(let s) = typeValue
    {
        ropeType = s
    } else {
        ropeType = "default"
    }

    switch ropeType {
    case "default":
        return RoPE(dimensions: dims, traditional: traditional, base: base, scale: 1.0)

    case "linear":
        let factor = scalingConfig?["factor"]?.asFloat() ?? 1.0
        let scale = 1.0 / factor
        return RoPE(dimensions: dims, traditional: traditional, base: base, scale: scale)

    case "llama3":
        return Llama3RoPE(
            dims: dims,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 2048,
            traditional: traditional,
            base: base,
            scalingConfig: scalingConfig ?? [:]
        )

    default:
        // Fall back to default RoPE for unsupported types
        return RoPE(dimensions: dims, traditional: traditional, base: base, scale: 1.0)
    }
}

// MARK: - Llama3 RoPE

/// Llama3 RoPE implementation with frequency adjustments.
private class Llama3RoPE: Module {
    let dims: Int
    let maxPositionEmbeddings: Int
    let traditional: Bool
    let _freqs: MLXArray

    init(
        dims: Int,
        maxPositionEmbeddings: Int = 2048,
        traditional: Bool = false,
        base: Float = 10000,
        scalingConfig: [String: StringOrNumber]
    ) {
        self.dims = dims
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.traditional = traditional

        let factor = scalingConfig["factor"]?.asFloat() ?? 1.0
        let lowFreqFactor = scalingConfig["low_freq_factor"]?.asFloat() ?? 1.0
        let highFreqFactor = scalingConfig["high_freq_factor"]?.asFloat() ?? 4.0
        let oldContextLen = scalingConfig["original_max_position_embeddings"]?.asFloat() ?? 8192

        let lowFreqWavelen = oldContextLen / lowFreqFactor
        let highFreqWavelen = oldContextLen / highFreqFactor

        let indices = MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32)
        var freqs = MLX.pow(MLXArray(base), indices / Float(dims))
        let wavelens = 2 * Float.pi * freqs

        // Apply factor to low frequency terms
        freqs = MLX.where(wavelens .> MLXArray(lowFreqWavelen), freqs * factor, freqs)

        // Identify medium frequency terms
        let isMediumFreq = (wavelens .> MLXArray(highFreqWavelen)) & (wavelens .< MLXArray(lowFreqWavelen))

        // Compute smooth factors for medium frequencies
        let smoothFactors = (oldContextLen / wavelens - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
        let smoothFreqs = freqs / ((1 - smoothFactors) / factor + smoothFactors)

        self._freqs = MLX.where(isMediumFreq, smoothFreqs, freqs)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        return MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: _freqs
        )
    }
}

// MARK: - Attention

private class Attention: Module {
    let args: Ministral3Configuration
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: Module

    init(_ args: Ministral3Configuration) {
        self.args = args

        let dim = args.hiddenSize
        self.nHeads = args.attentionHeads
        self.nKVHeads = args.kvHeads

        self.headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        // Initialize RoPE using rope_parameters
        let ropeTheta = args.ropeParameters?["rope_theta"]?.asFloat() ?? 10000.0
        self.rope = initializeRope(
            dims: headDim,
            base: ropeTheta,
            traditional: false,
            scalingConfig: args.ropeParameters,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, attnScale: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // Prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        // Apply RoPE
        let offset = cache?.offset ?? 0
        if let ropeModule = rope as? RoPE {
            queries = ropeModule(queries, offset: offset)
            keys = ropeModule(keys, offset: offset)
        } else if let llama3Rope = rope as? Llama3RoPE {
            queries = llama3Rope(queries, offset: offset)
            keys = llama3Rope(keys, offset: offset)
        }

        // Apply attention scaling
        queries = queries * attnScale

        // Update cache and compute attention
        let (cachedKeys, cachedValues): (MLXArray, MLXArray)
        if let cache = cache {
            (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        } else {
            (cachedKeys, cachedValues) = (keys, values)
        }

        // Compute scaled dot product attention
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask
        )

        let reshapedOutput = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return wo(reshapedOutput)
    }
}

// MARK: - MLP

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: Ministral3Configuration) {
        let dim = args.hiddenSize
        let hiddenDim = args.intermediateSize

        self._gate.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._down.wrappedValue = Linear(hiddenDim, dim, bias: false)
        self._up.wrappedValue = Linear(dim, hiddenDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return down(silu(gate(x)) * up(x))
    }
}

// MARK: - Transformer Block

private class TransformerBlock: Module {
    let numAttentionHeads: Int
    let hiddenSize: Int
    let useSliding: Bool

    @ModuleInfo(key: "self_attn") var attention: Attention
    @ModuleInfo(key: "mlp") var mlp: MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: Ministral3Configuration, useSliding: Bool = false) {
        self.numAttentionHeads = args.attentionHeads
        self.hiddenSize = args.hiddenSize
        self.useSliding = useSliding

        self._attention.wrappedValue = Attention(args)
        self._mlp.wrappedValue = MLP(args)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, attnScale: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = attention(inputLayerNorm(x), attnScale: attnScale, mask: mask, cache: cache)
        let h = x + r
        let mlpOut = mlp(postAttentionLayerNorm(h))
        let out = h + mlpOut
        return out
    }
}

// MARK: - Language Model (Inner)

private class Ministral3ModelInner: Module {
    let args: Ministral3Configuration
    let vocabularySize: Int
    let numHiddenLayers: Int
    let layerTypes: [String]
    let slidingWindow: Int?

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [TransformerBlock]
    let norm: RMSNorm

    // Indices for first full attention and sliding window attention layers
    let faIdx: Int
    let swaIdx: Int?

    init(_ args: Ministral3Configuration) {
        self.args = args
        self.vocabularySize = args.vocabularySize
        self.numHiddenLayers = args.hiddenLayers
        self.layerTypes = args.layerTypes
        self.slidingWindow = args.slidingWindow

        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        // Create transformer blocks with appropriate attention type
        self.layers = args.layerTypes.map { layerType in
            TransformerBlock(args, useSliding: layerType == "sliding_attention")
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        // Find the first full attention layer index
        self.faIdx = args.layerTypes.firstIndex(of: "full_attention") ?? 0

        // Find the first sliding window attention layer index
        self.swaIdx = args.layerTypes.firstIndex(of: "sliding_attention")

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]? = nil
    ) -> MLXArray {
        var h = embedTokens(inputs)

        let offset = cache?.first?.offset ?? 0

        // Create full attention mask
        let faMask = createAttentionMask(h: h, cache: cache?[faIdx])

        // Create sliding window attention mask
        let swaMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let swaIdx = swaIdx {
            swaMask = createAttentionMask(h: h, cache: cache?[swaIdx], windowSize: slidingWindow)
        } else {
            swaMask = .none
        }

        // Compute attention scale
        let llama4ScalingBeta = args.ropeParameters?["llama_4_scaling_beta"]?.asFloat() ?? 0.0
        let originalMaxPosEmbed = args.ropeParameters?["original_max_position_embeddings"]?.asInt() ?? args.maxPositionEmbeddings ?? 32768

        let attnScale = getLlama4AttentionScale(
            start: offset,
            stop: offset + inputs.dim(1),
            beta: llama4ScalingBeta,
            maxPositionEmbeddings: originalMaxPosEmbed
        ).asType(h.dtype)

        // Process through transformer layers
        for (i, layer) in layers.enumerated() {
            let mask = layer.useSliding ? swaMask : faMask
            h = layer(h, attnScale: attnScale, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Model

/// Ministral3 language model.
public class Ministral3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: Ministral3ModelInner
    fileprivate let args: Ministral3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Ministral3Configuration) {
        self.args = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Ministral3ModelInner(args)

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Remove unused precomputed rotary frequencies
        var sanitizedWeights = weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }

        // Handle tied embeddings
        if args.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        // Handle weight_scale_inv for quantized weights
        var newWeights: [String: MLXArray] = [:]
        for (key, value) in sanitizedWeights {
            if key.contains("weight_scale_inv") {
                let scaleInv = value
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = sanitizedWeights[weightKey] {
                    newWeights[weightKey] = weight * scaleInv
                }
            } else if key.contains("activation_scale") {
                continue
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }

        return newWeights.isEmpty ? sanitizedWeights : newWeights
    }

    /// Create appropriate caches for each layer type.
    ///
    /// Sliding window attention layers use RotatingKVCache,
    /// full attention layers use standard KVCacheSimple.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.useSliding, let slidingWindow = args.slidingWindow {
                return RotatingKVCache(maxSize: slidingWindow)
            } else {
                return KVCacheSimple()
            }
        }
    }
}

// MARK: - Configuration

public struct Ministral3Configuration: Codable, Sendable {
    var modelType: String = "ministral3"
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var headDimensions: Int?
    var maxPositionEmbeddings: Int?
    var kvHeads: Int
    var ropeParameters: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool = true
    var layerTypes: [String]
    var slidingWindow: Int?

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case headDimensions = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case kvHeads = "num_key_value_heads"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "ministral3"
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        ropeParameters = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true

        // Handle layer_types with default to all full_attention
        if let types = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = types
        } else {
            layerTypes = Array(repeating: "full_attention", count: hiddenLayers)
        }

        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
    }

    public init(
        modelType: String = "ministral3",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        rmsNormEps: Float,
        vocabularySize: Int,
        headDimensions: Int? = nil,
        maxPositionEmbeddings: Int? = nil,
        kvHeads: Int? = nil,
        ropeParameters: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true,
        layerTypes: [String]? = nil,
        slidingWindow: Int? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.headDimensions = headDimensions
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.kvHeads = kvHeads ?? attentionHeads
        self.ropeParameters = ropeParameters
        self.tieWordEmbeddings = tieWordEmbeddings
        self.layerTypes = layerTypes ?? Array(repeating: "full_attention", count: hiddenLayers)
        self.slidingWindow = slidingWindow
    }
}

// MARK: - LoRA

extension Ministral3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
