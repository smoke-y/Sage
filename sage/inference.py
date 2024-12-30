import os
import glob
from gemma import *
from sys import argv
from siglip import *
from safetensors import safe_open

#assert len(argv) > 2, "Image path and text required"
# imagePath = argv[1]
# text = argv[2]

MODEL_DIR = "paligemma-3b-pt-224"

config = PaliGemmaConfig(vision_config=SiglipConfig(), text_config=GemmaConfig())
safetensors_files = glob.glob(os.path.join(MODEL_DIR, "*.safetensors"))
tensors = {}
for safetensors_file in safetensors_files:
    with safe_open(safetensors_file, framework="pt", device="cpu") as f:
        for key in f.keys(): tensors[key] = f.get_tensor(key)