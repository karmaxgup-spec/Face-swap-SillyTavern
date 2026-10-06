"""
One-time model prep for the ST-FaceSwap extension.

    pip install onnx onnxconverter-common numpy
    python prepare_models.py inswapper_128.onnx

Produces (next to the input file):
  emap.bin                  512x512 float32 identity-embedding map, read by the extension
  inswapper_128_fp16.onnx   FP16 weights with FP32 inputs/outputs (keep_io_types) -> ~half the size

Then copy these into  ST-FaceSwap/models/  together with:
  det_2.5g.onnx        (SCRFD-2.5G, from the insightface buffalo_m pack)
  w600k_r50.onnx       (ArcFace, from the insightface buffalo_m pack)
  GFPGANv1.4.onnx      (optional enhancer)
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper

src = sys.argv[1] if len(sys.argv) > 1 else "inswapper_128.onnx"
model = onnx.load(src)

# insightface's InSwapper reads emap from the LAST initializer of the graph
emap = numpy_helper.to_array(model.graph.initializer[-1]).astype(np.float32)
assert emap.shape == (512, 512), f"unexpected emap shape {emap.shape}"
emap.tofile("emap.bin")
print("wrote emap.bin", emap.shape)

from onnxconverter_common import float16

model16 = float16.convert_float_to_float16(model, keep_io_types=True)
onnx.save(model16, "inswapper_128_fp16.onnx")
print("wrote inswapper_128_fp16.onnx")
