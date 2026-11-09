import json, os
base = MODELDIR
for f in ["tokenizer.json","model.onnx","config.json","special_tokens_map.json"]:
    p = os.path.join(base, f)
    print(f, os.path.getsize(p), "bytes")
tok = json.load(open(base+"/tokenizer.json"))
print("tokenizer model type:", tok["model"]["type"], "| vocab size:", len(tok["model"].get("vocab",{})))
cfg = json.load(open(base+"/config.json"))
print("hidden_size:", cfg.get("hidden_size"))
# onnx header sanity
with open(base+"/model.onnx","rb") as fh:
    print("onnx magic bytes:", fh.read(8)[:4] in (b"b\x0001", b"\x00") or "check")
import onnxruntime as ort
so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(base+"/model.onnx", so, providers=["CPUExecutionProvider"])
print("inputs:", [(i.name, i.shape) for i in sess.get_inputs()])
print("outputs:", [(o.name, o.shape) for o in sess.get_outputs()])
import numpy as np
ids = np.array([[101,1045,12008,102]], dtype=np.int64)
mask = np.array([[1,1,1,1]], dtype=np.int64)
out = sess.run(None, {i.name:(ids if 'input_ids' in i.name else mask) for i in sess.get_inputs()})
for name, arr in zip([o.name for o in sess.get_outputs()], out):
    print(name, arr.shape, "first-vec-dim=", arr.shape[-1])
