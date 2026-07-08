# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI(title="Chest X-ray Diagnosis API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DISEASE_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

model = models.densenet121(weights=None)
model.classifier = torch.nn.Linear(1024, 14)
model.load_state_dict(torch.load('models/densenet121_chestxray.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

@app.get("/")
def root():
    return {"status": "running", "model": "DenseNet121", "mean_auc": 0.8031}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = torch.sigmoid(model(input_tensor))
        probabilities = outputs[0].tolist()

    results = {name: round(prob, 4) for name, prob in zip(DISEASE_NAMES, probabilities)}
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    return {"predictions": sorted_results}