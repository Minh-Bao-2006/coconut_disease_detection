import gradio as gr
import torch, joblib, numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from config import *

DISPLAY = {
    "Bud_Root_Dropping": "Bud Root Dropping (Rụng mầm gốc)",
    "Bud_Rot":           "Bud Rot (Thối chồi)",
    "Gray_Leaf_Spot":    "Gray Leaf Spot (Đốm lá xám)",
    "Leaf_Rot":          "Leaf Rot (Thối lá)",
    "Stem_Bleeding":     "Stem Bleeding (Chảy nhựa thân)",
}

TRANSFORM = T.Compose([
    T.Resize(
256
), T.CenterCrop(
224
), T.ToTensor(),
    T.Normalize([
0.485
,
0.456
,
0.406
],[
0.229
,
0.224
,
0.225
]),
])

backbone = models.resnet18(weights="IMAGENET1K_V1")
backbone.fc = nn.Identity()
backbone.eval()
clf = joblib.load(f"{CHECKPOINTS}/svm_pipeline.pkl")

def predict(image: Image.Image):
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(
0
)
    with torch.no_grad():
        feat = backbone(tensor).numpy()
    probs = clf.predict_proba(feat)[
0
]
    return {DISPLAY[CLASSES[i]]: float(probs[i]) for i in range(
5
)}

with gr.Blocks(title="Coconut Disease Detector") as demo:
    gr.Markdown("""
    ## Coconut Tree Disease Detector
    Upload ảnh lá hoặc thân cây dừa để phát hiện bệnh.
    """)
    with gr.Row():
        inp = gr.Image(type="pil", label="Ảnh đầu vào")
        out = gr.Label(num_top_classes=
5
, label="Kết quả")
    gr.Button("Phân tích").click(predict, inp, out)

demo.launch(share=False) 