Example 1:

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'path/to/the/model.onnx'
model_quant = 'path/to/the/model.quant.onnx'
weight_type = QuantType.QInt8
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type)

import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader

# Load a calibration data reader
calibration_data = CalibrationDataReader(calibration_data_path="path/to/data")

# Perform static quantization
quantized_model = quantize_static(​
 model_input="model.onnx",​
 model_output="quantized_model_static.onnx",​
 calibration_data_reader=calibration_data,​
 weight_type=QuantType.QInt8​
)

Example 2:

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from auto_round import AutoRound

model_name = "microsoft/phi4"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

bits, group_size, sym = 4, 128, True
autoround = AutoRound(model, tokenizer, nsamples=128, iters=256, low_gpu_mem_usage=False, gradient_accumulate_steps=1, batch_size=8, bits=bits, group_size=group_size, sym=sym)
autoround.quantize()
output_dir = "phi4-Instruct-AutoRound-GPTQ-4bit"
autoround.save_quantized(output_dir, format='auto_gptq', inplace=True)


import torch
from transformers import AutoModelForCausalLM

model_name = "openai/whisper"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)


