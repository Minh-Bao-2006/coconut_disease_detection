import torch, joblib, numpy as np
import torchvision.models as models
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image

CLASSES = ["Bud Root Dropping", "Bud Rot",
           "Gray Leaf Spot", "Leaf Rot", "Stem Bleeding"]

TRANSFORM = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class Predictor:
    def __init__(self, clf_path="checkpoints/svm_pipeline.pkl",
                 backbone="resnet18"):
        m = models.resnet18(weights="IMAGENET1K_V1")
        m.fc = nn.Identity()
        self.backbone = m.eval()          
        self.clf = joblib.load(clf_path)
    
    def predict(self, img_path: str) -> dict:
        img = Image.open(img_path).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0)
        with torch.no_grad():
            feat = self.backbone(tensor).numpy()
        probs = self.clf.predict_proba(feat)[0]
        pred_idx = probs.argmax()
        return {
            "class": CLASSES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "all_probs": {CLASSES[i]: float(p) for i, p in enumerate(probs)}
        }

if __name__ == "__main__":
    p = Predictor()
    result = p.predict("test_image.jpg")
    print(f"Bệnh: {result['class']} ({result['confidence']*100:.1f}%)")
    for cls, prob in result["all_probs"].items():
        print(f"  {cls}: {prob*100:.1f}%")