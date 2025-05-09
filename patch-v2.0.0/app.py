import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models as torchvision_models  # Renamed import
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Dict, Optional
from PIL import Image
import uvicorn
import nest_asyncio
import warnings
import base64
from pydantic import BaseModel 
from contextlib import asynccontextmanager

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------
# 1. Core Analysis Engine (Enhanced)
# ---------------------
class AIModelHub:
    """Central model repository with improved error handling"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = self._load_models()

    def _load_models(self) -> Dict[str, torch.nn.Module]:
        """Safely load all models with error handling"""
        model_dict = {}
        try:
            model_dict["detection"] = torch.hub.load(
                'ultralytics/yolov5',
                'yolov5s',
                pretrained=True,
                force_reload=False
            ).to(self.device).eval()

            model_dict["segmentation"] = torchvision_models.segmentation.deeplabv3_resnet101(
                pretrained=True
            ).to(self.device).eval()

            model_dict["classification"] = torchvision_models.resnet50(
                pretrained=True
            ).to(self.device).eval()

        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")
        return model_dict

    def analyze(self, image: np.ndarray, task: str) -> Dict:
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")

        available_tasks = {
            "detection": self._run_detection,
            "segmentation": self._run_segmentation,
            "classification": self._run_classification,
            "features": self._extract_features,
            "preprocessing": self._run_preprocessing
        }

        if task == "full_analysis":
            return {t: available_tasks[t](image) for t in available_tasks}

        if task not in available_tasks:
            raise ValueError(f"Invalid task: {task}. Available: {list(available_tasks.keys())}")

        return {task: available_tasks[task](image)}

    def _run_detection(self, image: np.ndarray) -> List[Dict]:
        try:
            results = self.models["detection"](image)
            return [{
                "label": results.names[int(cls)],
                "confidence": float(conf),
                "bbox": box.tolist()
            } for box, conf, cls in zip(
                results.xyxy[0][:, :4],
                results.xyxy[0][:, 4],
                results.xyxy[0][:, 5]
            )]
        except Exception as e:
            return {"error": f"Detection failed: {str(e)}"}

    def _run_segmentation(self, image: np.ndarray) -> Dict:
        try:
            preprocess = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.models["segmentation"](img_tensor)['out'][0]
            mask = output.argmax(0).byte().cpu().numpy()
            return {
                "mask": mask.tolist(),
                "shape": mask.shape
            }
        except Exception as e:
            return {"error": f"Segmentation failed: {str(e)}"}

    def _run_classification(self, image: np.ndarray) -> List[Dict]:
        try:
            preprocess = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.models["classification"](img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top_probs, top_classes = torch.topk(probs, 5)
            return [{
                "label": str(top_classes[i].item()),
                "confidence": float(top_probs[i]),
                "class_idx": int(top_classes[i])
            } for i in range(top_probs.size(0))]
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}

    def _extract_features(self, image: np.ndarray) -> Dict:
        try:
            preprocess = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.models["classification"](img_tensor)
            return {
                "features": features.squeeze().cpu().tolist(),
                "dimension": len(features.squeeze())
            }
        except Exception as e:
            return {"error": f"Feature extraction failed: {str(e)}"}

    def _run_preprocessing(self, image: np.ndarray) -> Dict:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 100, 200)
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            stacked = np.stack([morphed]*3, axis=-1)  # Convert to 3-channel
            _, encoded = cv2.imencode('.jpg', cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))
            base64_image = base64.b64encode(encoded).decode('utf-8')
            return {"image_base64": base64_image, "shape": stacked.shape}
        except Exception as e:
            return {"error": f"Preprocessing failed: {str(e)}"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_hub
    model_hub = AIModelHub()
    yield

app = FastAPI(
    title="AI Image Analysis API",
    description="Perform detection, segmentation, classification and feature extraction",
    version="1.0.0",
    lifespan=lifespan
)

class AnalysisRequest(BaseModel):
    tasks: List[str] = ["full_analysis"]
    
@app.get("/healthz")
def health_check():
    return {"status": "ok"}

@app.post("/analyze", response_model=Dict[str, Dict])
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyze"),
    tasks: List[str] = ["full_analysis"]
):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")

        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(400, "Invalid image format")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = {}

        for task in tasks:
            try:
                results.update(model_hub.analyze(image, task))
            except Exception as e:
                results[task] = {"error": str(e)}

        return {"results": results}

    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

# ---------------------
# 3. Visualization Module (Enhanced)
# ---------------------
def generate_visual_report(image: np.ndarray, analysis: Dict) -> Optional[bytes]:
    try:
        if "detection" in analysis and "error" not in analysis["detection"]:
            for detection in analysis["detection"]:
                x1, y1, x2, y2 = map(int, detection["bbox"])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"{detection['label']} {detection['confidence']:.2f}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36,255,12),
                    2
                )

        if "segmentation" in analysis and "error" not in analysis["segmentation"]:
            mask = np.array(analysis["segmentation"]["mask"])
            mask = cv2.applyColorMap((mask*10).astype(np.uint8), cv2.COLORMAP_JET)
            image = cv2.addWeighted(image, 0.6, mask, 0.4, 0)

        _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return img_encoded.tobytes()
    except Exception:
        return None

# ---------------------
# 4. Server Configuration
# ---------------------
if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=9090,
        reload=False,
        workers=1,
        timeout_keep_alive=30
    )
