//
//  AfMoE.swift
//  LLM
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/afmoe.py
//

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct AfMoEConfiguration: Codable, Sendable {
    var modelType: String = "afmoe"
    var layerTypes: [String]
    var vocabularySize: Int = 200192
    var hiddenSize: Int = 2048
    var intermediateSize: Int = 6144
    var moeIntermediateSize: Int = 1024
    var hiddenLayers: Int = 32
    var attentionHeads: Int = 32
    var kvHeads: Int = 4
    var headDim: Int = 64
    var maxPositionEmbeddings: Int = 131072
    var rmsNormEps: Float = 1e-5
    var ropeTheta: Float = 10000
    var ropeScaling: [String: StringOrNumber]? = nil
    var tieWordEmbeddings: Bool = false
    // MoE config
    var numExperts: Int = 128
    var numExpertsPerToken: Int = 8
    var numSharedExperts: Int = 1
    var numDenseLayers: Int = 2
    var routeNorm: Bool = true
    var routeScale: Float = 2.826
    var scoreFunc: String = "sigmoid"
    var nGroup: Int = 1
    var topkGroup: Int = 1
    var slidingWindow: Int = 2048
    var mupEnabled: Bool = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case layerTypes = "layer_types"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case numDenseLayers = "num_dense_layers"
        case routeNorm = "route_norm"
        case routeScale = "route_scale"
        case scoreFunc = "score_func"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case slidingWindow = "sliding_window"
        case mupEnabled = "mup_enabled"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "afmoe"
        self.layerTypes = try container.decode([String].self, forKey: .layerTypes)
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 200192
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 128
        self.numExpertsPerToken =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 8
        self.numSharedExperts =
            try container.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 1
        self.numDenseLayers =
            try container.decodeIfPresent(Int.self, forKey: .numDenseLayers) ?? 2
        self.routeNorm = try container.decodeIfPresent(Bool.self, forKey: .routeNorm) ?? true
        self.routeScale = try container.decodeIfPresent(Float.self, forKey: .routeScale) ?? 2.826
        self.scoreFunc = try container.decodeIfPresent(String.self, forKey: .scoreFunc) ?? "sigmoid"
        self.nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        self.topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        self.slidingWindow =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 2048
        self.mupEnabled = try container.decodeIfPresent(Bool.self, forKey: .mupEnabled) ?? true
    }
}

// MARK: - Attention

private class Attention: Module {
    let args: AfMoEConfiguration
    let scale: Float
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let isLocalAttention: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "gate_proj") var gate: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE?

    init(_ args: AfMoEConfiguration, isLocalAttention: Bool = false) {
        self.args = args
        self.nHeads = args.attentionHeads
        self.nKVHeads = args.kvHeads
        self.headDim = args.headDim
        self.isLocalAttention = isLocalAttention

        self.scale = pow(Float(args.headDim), -0.5)

        _wq.wrappedValue = Linear(args.hiddenSize, nHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(args.hiddenSize, nKVHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(args.hiddenSize, nKVHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(nHeads * headDim, args.hiddenSize, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        _gate.wrappedValue = Linear(args.hiddenSize, nHeads * headDim, bias: false)

        if isLocalAttention {
            // Initialize RoPE for local attention layers
            if let ropeScaling = args.ropeScaling,
                ropeScaling["type"] == .string("linear"),
                let factor = ropeScaling["factor"]?.asFloat()
            {
                let ropeScale = 1.0 / factor
                self.rope = RoPE(
                    dimensions: headDim,
                    traditional: false,
                    base: args.ropeTheta,
                    scale: ropeScale
                )
            } else {
                self.rope = RoPE(
                    dimensions: headDim,
                    traditional: false,
                    base: args.ropeTheta
                )
            }
        } else {
            self.rope = nil
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // Reshape and transpose: [B, L, H, D] -> [B, H, L, D]
        queries = queries.reshaped(B, L, nHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        // Apply Q/K normalization
        queries = qNorm(queries)
        keys = kNorm(keys)

        // Apply RoPE for local attention
        if isLocalAttention, let rope = self.rope {
            if let cache = cache {
                queries = rope(queries, offset: cache.offset)
                keys = rope(keys, offset: cache.offset)
            } else {
                queries = rope(queries)
                keys = rope(keys)
            }
        }

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        // Apply gating
        let gateValues = sigmoid(gate(x))
        output = output * gateValues

        return wo(output)
    }
}

// MARK: - MLP

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE Router

private class MoERouter: Module {
    @ModuleInfo(key: "gate") var gate: Linear

    init(hiddenSize: Int, numExperts: Int) {
        _gate.wrappedValue = Linear(hiddenSize, numExperts, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        gate(x)
    }
}

// MARK: - AfMoE MoE Block

private class AfMoEMoE: Module, UnaryLayer {
    let args: AfMoEConfiguration
    let numExperts: Int
    let numExpertsPerToken: Int
    let routeNorm: Bool
    let routeScale: Float
    let scoreFunc: String
    let nGroup: Int
    let topkGroup: Int

    @ModuleInfo(key: "router") var router: MoERouter
    @ModuleInfo(key: "expert_bias") var expertBias: MLXArray
    @ModuleInfo(key: "experts") var experts: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: MLP?

    init(_ args: AfMoEConfiguration) {
        self.args = args
        self.numExperts = args.numExperts
        self.numExpertsPerToken = args.numExpertsPerToken
        self.routeNorm = args.routeNorm
        self.routeScale = args.routeScale
        self.scoreFunc = args.scoreFunc
        self.nGroup = args.nGroup
        self.topkGroup = args.topkGroup

        _router.wrappedValue = MoERouter(hiddenSize: args.hiddenSize, numExperts: args.numExperts)

        _expertBias.wrappedValue = MLXArray.zeros([args.numExperts])

        _experts.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )

        if args.numSharedExperts > 0 {
            let sharedIntermediateSize = args.moeIntermediateSize * args.numSharedExperts
            _sharedExperts.wrappedValue = MLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: sharedIntermediateSize
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = router(x)

        // Compute scores based on score function
        let scores: MLXArray
        if scoreFunc == "sigmoid" {
            scores = sigmoid(gates.asType(.float32))
        } else {
            scores = softmax(gates.asType(.float32), axis: -1)
        }

        // Add expert bias for selection
        var selectionScores = scores + expertBias

        // Group-based expert selection if nGroup > 1
        if nGroup > 1 {
            // Reshape to groups: [B, L, E] -> [B, L, G, E/G]
            var groupScores = selectionScores.reshaped(
                selectionScores.dim(0), selectionScores.dim(1), nGroup, -1)
            // Get top-2 scores per group and sum
            let topKGroup = top(groupScores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            // Zero out bottom groups (get indices of k smallest groups)
            let k = nGroup - topkGroup
            let groupIdx = argPartition(topKGroup, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
            groupScores = putAlong(groupScores, groupIdx, values: MLXArray(0.0), axis: -2)
            // Flatten back
            selectionScores = flattened(groupScores, start: -2, end: -1)
        }

        // Select top-k experts
        let k = numExpertsPerToken
        let inds = argPartition(-selectionScores, kth: k - 1, axis: -1)[.ellipsis, ..<k]

        var selectedScores = takeAlong(scores, inds, axis: -1)

        // Normalize selected scores
        if routeNorm && numExpertsPerToken > 1 {
            let denominator = selectedScores.sum(axis: -1, keepDims: true)
            selectedScores = selectedScores / denominator
        }

        // Apply route scale
        selectedScores = selectedScores * routeScale

        // Compute expert outputs
        var y = experts(x, inds)
        y = (y * selectedScores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)

        // Add shared expert output
        if let sharedExperts = sharedExperts {
            y = y + sharedExperts(x)
        }

        return y
    }
}

// MARK: - Decoder Layer

private class AfMoEDecoderLayer: Module {
    let args: AfMoEConfiguration
    let layerIdx: Int
    let useSliding: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_mlp_layernorm") var preMlpLayerNorm: RMSNorm
    @ModuleInfo(key: "post_mlp_layernorm") var postMlpLayerNorm: RMSNorm

    fileprivate let mlp: UnaryLayer

    init(_ args: AfMoEConfiguration, layerIdx: Int, useSliding: Bool = false) {
        self.args = args
        self.layerIdx = layerIdx
        self.useSliding = useSliding

        _selfAttn.wrappedValue = Attention(args, isLocalAttention: useSliding)

        // Use dense MLP for early layers, MoE for the rest
        if layerIdx < args.numDenseLayers {
            self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        } else {
            self.mlp = AfMoEMoE(args)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preMlpLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postMlpLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        // Pre-norm attention with post-norm residual
        var r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        r = postAttentionLayerNorm(r)
        let h = x + r

        // Pre-norm MLP with post-norm residual
        r = mlp(preMlpLayerNorm(h))
        r = postMlpLayerNorm(r)

        return h + r
    }
}

// MARK: - AfMoE Model Inner

private class AfMoEModelInner: Module {
    let args: AfMoEConfiguration
    let slidingWindow: Int
    let mupEnabled: Bool
    let hiddenSize: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [AfMoEDecoderLayer]
    let norm: RMSNorm

    // Indices for full attention and sliding window attention layers
    let faIdx: Int
    let swaIdx: Int?

    init(_ args: AfMoEConfiguration) {
        self.args = args
        self.slidingWindow = args.slidingWindow
        self.mupEnabled = args.mupEnabled
        self.hiddenSize = args.hiddenSize

        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        // Build layers based on layer_types
        var layerList: [AfMoEDecoderLayer] = []
        for (idx, layerType) in args.layerTypes.enumerated() {
            let useSliding = layerType == "sliding_attention"
            layerList.append(AfMoEDecoderLayer(args, layerIdx: idx, useSliding: useSliding))
        }
        self.layers = layerList

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        // Find indices for attention mask creation
        if let idx = args.layerTypes.firstIndex(of: "full_attention") {
            self.faIdx = idx
        } else {
            self.faIdx = 0
        }

        self.swaIdx = layers.firstIndex(where: { $0.useSliding })
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        // Apply mup scaling
        if mupEnabled {
            h = h * sqrt(Float(hiddenSize))
        }

        // Create attention masks
        let faMask = createAttentionMask(h: h, cache: cache?[faIdx])

        let swaMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let swaIdx = swaIdx {
            swaMask = createAttentionMask(h: h, cache: cache?[swaIdx], windowSize: slidingWindow)
        } else {
            swaMask = .none
        }

        for (i, layer) in layers.enumerated() {
            let mask = layer.useSliding ? swaMask : faMask
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - AfMoE Model

public class AfMoEModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let slidingWindow: Int

    fileprivate let model: AfMoEModelInner
    let configuration: AfMoEConfiguration
    fileprivate let layerUsesSliding: [Bool]

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: AfMoEConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0..<args.hiddenLayers).map { _ in args.kvHeads }
        self.slidingWindow = args.slidingWindow
        self.model = AfMoEModelInner(args)

        // Track which layers use sliding attention
        self.layerUsesSliding = args.layerTypes.map { $0 == "sliding_attention" }

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead = lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights

        // Remove unused precomputed rotary freqs
        sanitizedWeights = sanitizedWeights.filter { !$0.key.contains("rotary_emb.inv_freq") }

        // Remove lm_head if tied embeddings
        if configuration.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        // Stack expert weights for SwitchGLU
        for l in 0..<configuration.hiddenLayers {
            if l < configuration.numDenseLayers {
                continue
            }
            let prefix = "model.layers.\(l)"
            for n in ["up_proj", "down_proj", "gate_proj"] {
                for k in ["weight", "scales", "biases"] {
                    if sanitizedWeights["\(prefix).mlp.experts.0.\(n).\(k)"] != nil {
                        let toJoin = (0..<configuration.numExperts).map { e in
                            sanitizedWeights.removeValue(forKey: "\(prefix).mlp.experts.\(e).\(n).\(k)")!
                        }
                        sanitizedWeights["\(prefix).mlp.experts.\(n).\(k)"] = MLX.stacked(toJoin)
                    }
                }
            }
        }

        return sanitizedWeights
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        // Create cache based on layer type (rotating for sliding attention, simple for full attention)
        return layerUsesSliding.map { usesSliding in
            if usesSliding {
                RotatingKVCache(maxSize: slidingWindow)
            } else {
                KVCacheSimple()
            }
        }
    }
}

// MARK: - LoRA

extension AfMoEModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
