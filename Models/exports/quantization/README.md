# Quantization Exports

To export quantized version of the model use `autosteer_2.0_ptq.py` script

```bash
  python3 quantize_ptq.py -q -d <DATASET_PATH> -fp32 <PYTORCH_MODEL_PATH> -int8onnx <QUANTIZED_ONNX_MODEL_PATH>
```