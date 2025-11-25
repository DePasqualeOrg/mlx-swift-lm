# Missing Cache Functionality in MLX Swift Port

This document provides a detailed comparison of KV cache implementations between Python mlx-lm and Swift mlx-swift-lm, highlighting missing features and functionality.

## Overview

The Python mlx-lm has a sophisticated caching system with multiple cache types, batch processing support, and advanced features. The Swift port has a solid foundation but is missing several key components.

**Python File**: `/mlx_lm/models/cache.py` (1199 lines)
**Swift File**: `/Libraries/MLXLMCommon/KVCache.swift` (1566 lines)

---

## Cache Types Comparison

### ✅ Implemented in Both

| Cache Type | Python | Swift | Notes |
|------------|--------|-------|-------|
| **KVCache/KVCacheSimple** | ✅ | ✅ | Basic cache with dynamic growth |
| **RotatingKVCache** | ✅ | ✅ | Sliding window attention cache |
| **QuantizedKVCache** | ✅ | ✅ | Memory-efficient quantized cache |
| **ChunkedKVCache** | ✅ | ✅ | Chunk-based processing |
| **ArraysCache** | ✅ | ✅ | Base for array storage |
| **MambaCache** | ✅ | ✅ | State space model cache |
| **CacheList** | ✅ | ✅ | Composite cache container |

### ❌ Missing in Swift

| Cache Type | Description | Use Case | Priority |
|------------|-------------|----------|----------|
| **ConcatenateKVCache** | Simple concatenation-based cache | Mock/testing, large block processing | ⭐⭐ |
| **BatchKVCache** | Left-padded batch processing | Efficient batch inference | ⭐⭐⭐⭐⭐ |
| **BatchRotatingKVCache** | Rotating cache with batch support | Batch + sliding window | ⭐⭐⭐⭐⭐ |

---

## Detailed Missing Cache Types

### 1. ConcatenateKVCache ❌

**Python Location**: Lines 176-219

**Purpose**: Simplest KV cache that just concatenates keys/values without pre-allocation.

**Key Features**:
- No step-based pre-allocation (unlike KVCache)
- Pure concatenation approach
- Useful as mock cache or when processing large blocks at once
- Trimmable

**Implementation**:
```python
class ConcatenateKVCache(_BaseCache):
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=-2)
            self.values = mx.concatenate([self.values, values], axis=-2)
        self.offset = self.keys.shape[-2]
        return self.keys, self.values
```

**Use Cases**:
- Testing and mocking
- Processing very large prompts where pre-allocation overhead is significant
- Scenarios where you don't know final size ahead of time

**Priority**: ⭐⭐ (Low - mostly for compatibility and specific edge cases)

---

### 2. BatchKVCache ❌

**Python Location**: Lines 699-859

**Purpose**: High-performance batched inference with left-padding support.

**Key Features**:
- **Left-padding aware**: Handles variable-length sequences in batch
- **Per-sequence offsets**: Tracks offset per batch item
- **Dynamic padding management**: `prepare()` and `finalize()` for padding
- **Batch operations**: `filter()`, `extend()`, `extract()`, `merge()`
- **Rolling for alignment**: Uses `dynamic_roll()` for right-padding
- **Efficient masking**: Creates batch-aware causal masks

**Critical Methods**:

1. **`prepare(left_padding, lengths, right_padding)`**:
   - Adjusts padding before processing
   - Handles right-padding via internal tracking

2. **`finalize()`**:
   - Rolls cache to correct alignment after processing
   - Removes right-padding artifacts

3. **`filter(batch_indices)`**:
   - Keeps only specified batch indices
   - Optimizes by removing common left padding

4. **`extend(other)`**:
   - Combines two batch caches
   - Pads to common alignment

5. **`extract(idx)`**:
   - Extracts single sequence as regular KVCache
   - Useful for beam search or individual processing

6. **`merge(caches)`** (classmethod):
   - Merges multiple single caches into BatchKVCache
   - Creates unified batch cache from individual caches

**Example Usage**:
```python
# Create batch cache with left padding
cache = BatchKVCache(left_padding=[1, 3, 0])  # 3 sequences

# Prepare with right padding for variable lengths
cache.prepare(right_padding=[0, 1, 2])

# Process batch
keys, values = cache.update_and_fetch(batch_keys, batch_values)

# Finalize to correct alignment
cache.finalize()

# Filter to keep only best candidates
cache.filter(mx.array([0, 2]))  # Keep 1st and 3rd sequences
```

**Priority**: ⭐⭐⭐⭐⭐ (Critical - essential for efficient batch inference)

**Complexity**: High - requires careful padding management and Metal-specific optimizations

---

### 3. BatchRotatingKVCache ❌

**Python Location**: Lines 886-1198

**Purpose**: Combines batch processing with rotating/sliding window cache.

**Key Features**:
- **All BatchKVCache features** plus **sliding window attention**
- **Per-batch rotation tracking**: Different sequences can rotate independently
- **Temporal ordering**: `_temporal_order()` to handle rotated state
- **Right-padding protection**: Prevents valid tokens from being evicted by padding
- **Complex masking**: Handles both batch padding and rotation

**Critical Methods**:

1. **`_temporal_order()`**:
   - Rearranges rotated cache back to temporal order
   - Essential for correct cache management

2. **`_update_concat(keys, values)`**:
   - Handles multi-token updates (like prompt prefill)
   - Manages cache trimming with batch awareness

3. **`_update_in_place(keys, values)`**:
   - Single-token generation updates
   - Handles rotation at max_size boundary

4. **`make_mask(N, window_size, return_array)`**:
   - Creates batch-aware sliding window masks
   - Accounts for rotation state and padding

5. **`extract(idx)`**:
   - Extracts single sequence as RotatingKVCache
   - Handles rotation state correctly

6. **`merge(caches)`** (classmethod):
   - Merges multiple RotatingKVCaches into batch version
   - Validates consistent max_size

**Example Usage**:
```python
# Create batch rotating cache
cache = BatchRotatingKVCache(
    max_size=2048,
    left_padding=[0, 5, 2]
)

# Process long sequences with sliding window
for chunk in long_sequences:
    keys, values = cache.update_and_fetch(chunk_keys, chunk_values)
    # Automatically rotates when reaching max_size
```

**Priority**: ⭐⭐⭐⭐⭐ (Critical - essential for efficient long-context batch inference)

**Complexity**: Very High - most complex cache type, requires deep understanding of both batching and rotation

---

## Utility Functions

### ✅ Implemented in Both

| Function | Python | Swift | Notes |
|----------|--------|-------|-------|
| `make_prompt_cache()` | ✅ | ✅ | Factory function |
| `save_prompt_cache()` | ✅ | ✅ | Serialization |
| `load_prompt_cache()` | ✅ | ✅ | Deserialization |
| `can_trim_prompt_cache()` | ✅ | ✅ | Check trimmability |
| `trim_prompt_cache()` | ✅ | ✅ | Trim tokens |
| `cache_length()` | ✅ | Partial | Python: explicit function; Swift: property |

### ❌ Missing in Swift

| Function | Description | Priority |
|----------|-------------|----------|
| `dynamic_roll()` | Batch-aware rolling operation | ⭐⭐⭐⭐⭐ |

---

## Detailed Missing Utilities

### dynamic_roll() ❌

**Python Location**: Lines 690-697

**Purpose**: Roll arrays along an axis with per-element shift amounts (batch-aware rolling).

**Implementation**:
```python
def dynamic_roll(x, shifts, axis):
    n = x.shape[axis]
    expand_shifts = (...,) + (None,) * (x.ndim - axis)
    expand_indices = expand_shifts[:-1]
    idx = (mx.arange(n)[expand_indices] - shifts[expand_shifts]) % n
    rolled = mx.take_along_axis(x, idx, axis=axis)
    return rolled
```

**Use Case**:
Essential for BatchKVCache and BatchRotatingKVCache to handle per-sequence padding adjustments.

**Example**:
```python
# Roll each sequence by different amounts
cache = mx.random.normal((3, 8, 100, 64))  # [B, H, L, D]
shifts = mx.array([5, 10, 0])  # Different shift per batch item
rolled = dynamic_roll(cache, shifts[:, None], axis=2)
```

**Priority**: ⭐⭐⭐⭐⭐ (Critical - required for batch cache implementations)

---

## Feature Comparison

### Cache Management

| Feature | Python | Swift | Notes |
|---------|--------|-------|-------|
| **State serialization** | ✅ | ✅ | Both support safetensors |
| **Meta-state tracking** | ✅ | ✅ | Cache-specific metadata |
| **Cross-platform compat** | ✅ | ✅ | Python ↔ Swift cache files |
| **Trimming support** | ✅ | ✅ | Most caches trimmable |
| **Quantization** | ✅ | ✅ | Both have QuantizedKVCache |
| **Batch operations** | ✅ | ❌ | **Missing BatchKVCache** |
| **Dynamic padding** | ✅ | ❌ | **Missing prepare/finalize** |
| **Cache merging** | ✅ | ❌ | **Missing merge operations** |
| **Cache extraction** | ✅ | ❌ | **Missing extract operations** |

### Mask Generation

| Feature | Python | Swift | Notes |
|---------|--------|-------|-------|
| **Basic causal masks** | ✅ | ✅ | Standard causal attention |
| **Windowed masks** | ✅ | ✅ | Sliding window support |
| **Batch-aware masks** | ✅ | ❌ | **Missing BatchKVCache masks** |
| **Left-padding masks** | ✅ | Partial | Swift has limited support |
| **Rotation-aware masks** | ✅ | ✅ | RotatingKVCache handles this |
| **Combined batch+rotation** | ✅ | ❌ | **Missing BatchRotatingKVCache** |

### Memory Management

| Feature | Python | Swift | Notes |
|---------|--------|-------|-------|
| **Step-based pre-allocation** | ✅ | ✅ | 256-step default |
| **Dynamic growth** | ✅ | ✅ | Automatic expansion |
| **Quantized storage** | ✅ | ✅ | 4/8-bit support |
| **Rotation eviction** | ✅ | ✅ | RotatingKVCache |
| **Batch memory efficiency** | ✅ | ❌ | **Missing batch caches** |
| **Chunked processing** | ✅ | ✅ | ChunkedKVCache |

---

## Implementation Differences

### 1. Cache Protocol/Interface

**Python**:
```python
class _BaseCache:
    @property
    def state(self): ...

    @property
    def meta_state(self): ...

    def is_trimmable(self): ...

    def __len__(self): ...

    def __bool__(self): return True  # Special override
```

**Swift**:
```swift
public protocol KVCache: Evaluatable {
    var offset: Int { get }
    var maxSize: Int? { get }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    var state: [MLXArray] { get set }
    var metaState: [String] { get set }
    var isTrimmable: Bool { get }
    func trim(_ n: Int) -> Int
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MaskMode
}
```

**Key Differences**:
- Swift uses protocol (more flexible)
- Swift adds explicit `makeMask()` to protocol (Python: per-cache method)
- Python has `__len__` and `__bool__` operators
- Swift has `innerState()` for evaluation tracking

### 2. Quantization Approach

**Python**: Returns quantized tuples `(wq, scales, biases)` directly

**Swift**: Has dedicated `QuantizedKVCacheProtocol` with:
```swift
protocol QuantizedKVCacheProtocol: KVCache {
    var groupSize: Int { get }
    var bits: Int { get }
    func updateQuantized(keys: MLXArray, values: MLXArray)
        -> ((MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?))
}
```

**Swift Advantage**: Type-safe quantized path, no runtime type checking needed

### 3. Mask Creation

**Python**: Returns `None`, `"causal"`, or `MLXArray`

**Swift**: Uses enum for type safety:
```swift
enum ScaledDotProductAttentionMaskMode {
    case none
    case causal
    case array(MLXArray)
    case arrays([MLXArray])
}
```

**Swift Advantage**: Compile-time safety, clearer API

---

## Missing Advanced Features

### 1. Batch Operations ❌

**Python has, Swift missing**:

```python
# Filter to keep specific batch indices
cache.filter(batch_indices=[0, 2, 5])

# Extend cache with another cache
cache.extend(other_cache)

# Extract single sequence
single_cache = cache.extract(idx=0)

# Merge multiple caches
BatchKVCache.merge([cache1, cache2, cache3])
```

**Priority**: ⭐⭐⭐⭐⭐

**Use Cases**:
- Beam search (filter/extract)
- Batch processing optimization
- Dynamic batching
- Speculative decoding

### 2. Padding Management ❌

**Python has, Swift missing**:

```python
# Prepare cache for processing
cache.prepare(
    left_padding=[1, 2, 0],
    right_padding=[0, 1, 2],
    lengths=[10, 20, 15]
)

# Process...

# Finalize to correct alignment
cache.finalize()
```

**Priority**: ⭐⭐⭐⭐⭐

**Use Cases**:
- Variable-length sequences
- Efficient batch packing
- Memory alignment optimization

### 3. Temporal Ordering ❌

**Python has, Swift missing** (in BatchRotatingKVCache):

```python
def _temporal_order(self):
    """Rearrange cache into temporal order"""
    if self.rotated:
        self.keys = mx.roll(self.keys, -self._idx, axis=2)
        self.values = mx.roll(self.values, -self._idx, axis=2)
        self._idx = self.keys.shape[2]
        self.rotated = False
```

**Priority**: ⭐⭐⭐⭐

**Use Cases**:
- Correct state serialization of rotated caches
- Cache merging/extending
- Debugging and inspection

### 4. Right-Padding Protection ❌

**Python has, Swift missing** (in BatchRotatingKVCache):

```python
# Roll right sequences that are padded to prevent evicting valid tokens
if self._lengths is not None:
    roll = mx.maximum(0, self.offset - self._lengths)
    self.keys = dynamic_roll(self.keys, roll[:, None], axis=2)
    self.values = dynamic_roll(self.values, roll[:, None], axis=2)
```

**Priority**: ⭐⭐⭐⭐

**Use Cases**:
- Correct handling of right-padded batches
- Preserving valid tokens during rotation
- Long-context batch processing

---

## Performance Implications

### Memory Efficiency

| Scenario | Python | Swift | Impact |
|----------|--------|-------|--------|
| **Single sequence** | Efficient | Efficient | ✅ Equal |
| **Batch (same length)** | Efficient | Inefficient* | ⚠️ Without BatchKVCache, must use separate caches |
| **Batch (variable length)** | Very Efficient | Very Inefficient* | ❌ Major waste with padding |
| **Batch + long context** | Very Efficient | Very Inefficient* | ❌ Compounds memory issues |

*Swift must create separate caches per sequence without batch support

### Computational Efficiency

| Operation | Python | Swift | Impact |
|-----------|--------|-------|--------|
| **Single generation** | Fast | Fast | ✅ Equal |
| **Batch generation** | Fast | Slow* | ⚠️ No batch optimization |
| **Beam search** | Fast | Slow* | ❌ No extract/filter operations |
| **Dynamic batching** | Fast | N/A | ❌ Not possible |

*Without batch caches, operations require manual management

---

## Use Case Impact Analysis

### ✅ Works Well in Swift

1. **Single-sequence generation**
   - Standard chat/completion
   - Long context with RotatingKVCache
   - Quantized inference

2. **Fixed-batch generation**
   - Can use multiple separate caches
   - Inefficient but functional

### ⚠️ Limited in Swift

1. **Variable-length batches**
   - Requires manual padding to max length
   - Wastes memory
   - Inefficient

2. **Beam search**
   - Must manually manage cache copies
   - No efficient extract/filter

### ❌ Not Possible in Swift

1. **Efficient batch inference**
   - Can't handle left-padding properly
   - Memory waste scales with batch size

2. **Dynamic batching**
   - No merge/extend operations
   - Can't adjust batch composition dynamically

3. **Speculative decoding (batch)**
   - Requires batch cache operations
   - Critical for efficiency

4. **Production batch serving**
   - Left-padding handling essential
   - Currently requires workarounds

---

## Implementation Roadmap

### Phase 1: Critical Foundations ⭐⭐⭐⭐⭐

1. **Implement `dynamic_roll()`**
   - Required for all batch caches
   - Test with various shapes and shifts
   - Optimize for Metal

2. **Add `BatchKVCache`**
   - Start with basic update/fetch
   - Add padding support
   - Implement masking

### Phase 2: Batch Operations ⭐⭐⭐⭐⭐

3. **Add batch operations to `BatchKVCache`**
   - `filter()`
   - `extend()`
   - `extract()`
   - `merge()` classmethod

4. **Test suite for batch operations**
   - Variable-length sequences
   - Padding edge cases
   - Memory efficiency validation

### Phase 3: Advanced Batching ⭐⭐⭐⭐

5. **Implement `BatchRotatingKVCache`**
   - Temporal ordering
   - Right-padding protection
   - Complex masking

6. **Optimize Metal kernels**
   - Profile batch operations
   - Optimize rolling operations
   - Reduce memory copies

### Phase 4: Compatibility ⭐⭐

7. **Add `ConcatenateKVCache`**
   - For testing/compatibility
   - Alternative for specific use cases

8. **Cross-platform testing**
   - Python ↔ Swift cache interchange
   - Edge case validation

---

## Testing Requirements

### Unit Tests Needed

1. **BatchKVCache**:
   ```swift
   testBatchCacheWithLeftPadding()
   testBatchCachePrepareFinalize()
   testBatchCacheFilter()
   testBatchCacheExtend()
   testBatchCacheExtract()
   testBatchCacheMerge()
   ```

2. **BatchRotatingKVCache**:
   ```swift
   testBatchRotatingTemporalOrder()
   testBatchRotatingWithPadding()
   testBatchRotatingMasking()
   testBatchRotatingRightPaddingProtection()
   ```

3. **dynamic_roll()**:
   ```swift
   testDynamicRollSimple()
   testDynamicRollBatched()
   testDynamicRollNegativeShifts()
   testDynamicRollZeroShifts()
   ```

### Integration Tests Needed

1. **Batch generation** with variable-length prompts
2. **Beam search** with cache manipulation
3. **Long-context batch** with rotation
4. **Padding edge cases**
5. **Memory efficiency** benchmarks

### Compatibility Tests

1. **Save/Load** BatchKVCache with Python
2. **Cross-platform** cache interchange
3. **Quantization** of batch caches

---

## API Design Considerations

### Swift-Specific Enhancements

1. **Type Safety**:
   ```swift
   // Use enums for padding types
   enum PaddingConfig {
       case left([Int])
       case right([Int])
       case both(left: [Int], right: [Int])
   }

   cache.prepare(padding: .left([1, 2, 0]))
   ```

2. **Async/Actor Support**:
   ```swift
   actor BatchCacheManager {
       var cache: BatchKVCache

       func update(keys: MLXArray, values: MLXArray) async -> (MLXArray, MLXArray)
   }
   ```

3. **Combine/AsyncSequence**:
   ```swift
   extension BatchKVCache {
       func generationStream() -> AsyncStream<(MLXArray, MLXArray)>
   }
   ```

### Metal Optimization Opportunities

1. **Custom kernels** for `dynamic_roll()`
2. **Fused operations** for padding + roll
3. **Memory pooling** for batch caches
4. **Async compute** for cache updates

---

## Documentation Needs

### API Documentation

1. Comprehensive DocC documentation for:
   - `BatchKVCache` with usage examples
   - `BatchRotatingKVCache` with complex examples
   - Padding strategies and best practices

2. Migration guide from single-cache to batch-cache

3. Performance tuning guide

### Examples

1. **Basic batch inference**:
   ```swift
   // Example: Batch generation with variable lengths
   let cache = BatchKVCache(leftPadding: [0, 5, 3])
   // ... generate with batch
   ```

2. **Beam search**:
   ```swift
   // Example: Beam search with cache extraction
   let candidates = beams.map { cache.extract($0) }
   ```

3. **Dynamic batching**:
   ```swift
   // Example: Merging caches from multiple sources
   let batchCache = BatchKVCache.merge(individualCaches)
   ```

---

## Comparison Summary

### What Swift Has ✅

- ✅ Solid foundation with 7 cache types
- ✅ Quantization support
- ✅ Rotating/sliding window caches
- ✅ Serialization/deserialization
- ✅ Type-safe mask API
- ✅ Good single-sequence performance

### What Swift Needs ❌

- ❌ **BatchKVCache** (Critical)
- ❌ **BatchRotatingKVCache** (Critical)
- ❌ **`dynamic_roll()`** utility (Critical)
- ❌ Batch operations (filter, extend, extract, merge)
- ❌ Padding management (prepare, finalize)
- ❌ ConcatenateKVCache (Low priority)

### Priority Summary

**Must-Have** (⭐⭐⭐⭐⭐):
1. dynamic_roll() - Foundation for everything
2. BatchKVCache - Core batch functionality
3. Batch operations - filter, extend, extract, merge
4. BatchRotatingKVCache - Long context batching

**Should-Have** (⭐⭐⭐):
- Comprehensive testing suite
- Performance optimizations
- Documentation and examples

**Nice-to-Have** (⭐⭐):
- ConcatenateKVCache
- Additional Metal optimizations

---

## Conclusion

The Swift port has excellent single-sequence cache support but lacks critical batch processing capabilities. The missing `BatchKVCache` and `BatchRotatingKVCache` are major limitations for:

- Production batch inference servers
- Efficient variable-length batch processing
- Beam search and advanced decoding strategies
- Memory-efficient long-context batching

**Estimated Implementation Effort**:
- `dynamic_roll()`: 1-2 days
- `BatchKVCache`: 1 week
- Batch operations: 3-4 days
- `BatchRotatingKVCache`: 1-2 weeks
- Testing & optimization: 1 week

**Total**: ~4-5 weeks for complete batch cache support

---

**Last Updated**: 2025-11-24
**Python Version**: mlx-lm cache.py (1199 lines)
**Swift Version**: mlx-swift-lm KVCache.swift (1566 lines)
