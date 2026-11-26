# Compression Analysis: neural_network

- **Model type**: NeuralNetwork
- **Parameters**: 131
- **Layers**: 4
- **Estimated size**: 0.00 MB
- **Sparsity**: 0.0%

## Recommendations

### Quantization (int8_dynamic)

Convert weights from fp32 to int8. 4x size reduction with minimal accuracy loss.

- **Compression ratio**: 4.00x
- **Accuracy impact**: -0.50%
- **Difficulty**: easy
- **Risk score**: 0.05/1.0
- **Pros**: 4x size reduction, Minimal accuracy loss (~0.5%), Faster CPU inference, No retraining needed
- **Cons**: Slight precision loss, Needs int8-capable hardware for speedup
- **Target layers**: fc1.weight, fc1.bias, fc2.weight, fc2.bias

```python
import qai
quantized = qai.quantize_dynamic(model, dtype='q8')
# or:
from qai.quantization import Quantizer
q = Quantizer(format='int8')
quantized = q.apply(model)
```

### Early_Exit (confidence_based)

Add early exit branches after middle layers. Easy inputs exit early, hard inputs use full network.

- **Compression ratio**: 1.50x
- **Accuracy impact**: -0.50%
- **Difficulty**: medium
- **Risk score**: 0.15/1.0
- **Pros**: Dynamic compute (faster for easy inputs), No accuracy loss for easy inputs, Good for edge deployment
- **Cons**: Added complexity, Hard inputs still need full compute, Latency variance
- **Target layers**: fc1.bias, fc2.weight

```python
from qai.compression import EarlyExitWrapper
ee = EarlyExitWrapper(exit_layers=[1, 2], confidence_threshold=0.9)
fast_model = ee.wrap(model)
```

## Best Strategy

**quantization** (int8_dynamic) — 4.00x compression with -0.50% accuracy impact.