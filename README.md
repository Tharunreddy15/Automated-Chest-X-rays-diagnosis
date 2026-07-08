# Automated Chest X-ray Diagnosis

A deep learning system that detects 14 thoracic diseases from chest X-ray images using a DenseNet121 model, served as a REST API with FastAPI and Docker.

## What it does

Upload a frontal chest X-ray and the model returns probability scores for 14 conditions: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, and Pneumothorax.

Trained on the NIH ChestX-ray14 dataset (112,120 images), achieving a mean AUC-ROC of 0.8031.

## Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- [VS Code](https://code.visualstudio.com/) (optional, for editing)
- The trained model file (see step 2 below)

## Setup

### 1. Clone the repository

Open VS Code, then open a terminal (Terminal -> New Terminal) and run:

```bash
git clone https://github.com/Tharunreddy15/Automated-Chest-X-rays-diagnosis.git
cd Automated-Chest-X-rays-diagnosis
```

### 2. Add the model file

The trained model weights are not included in this repository due to file size. Download them and place the file at `models/densenet121_chestxray.pth`:

```bash
mkdir -p models
```

Download `densenet121_chestxray.pth` from Hugging Face Hub and move it into the `models/` folder:
https://huggingface.co/Tharunreddy1520/chestxray-densenet121

### 3. Build the Docker image

```bash
docker build -t chest-xray-api .
```

This installs all dependencies inside a container, so you don't need Python or PyTorch installed on your machine.

### 4. Run the container

```bash
docker run -p 8000:8000 chest-xray-api
```

Leave this terminal running. You should see `Uvicorn running on http://0.0.0.0:8000`.

## Using the API

### Option A - Interactive browser page (easiest)

Open this URL in your browser:

```
http://localhost:8000/docs
```

Click on the `/predict` endpoint -> **Try it out** -> upload a chest X-ray image -> **Execute**. The prediction appears on the page.

### Option B - Command line

Open a **second** terminal (keep the container running in the first) and run:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/your/xray.png"
```

Replace `path/to/your/xray.png` with the path to a chest X-ray image on your computer.

### Example response

```json
{
  "predictions": {
    "Cardiomegaly": 0.9981,
    "Fibrosis": 0.8162,
    "Effusion": 0.6203,
    "Atelectasis": 0.5609
  }
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Status check |
| `/health` | GET | Health check |
| `/predict` | POST | Upload an X-ray, get disease probabilities |

## Project Structure

```
.
├── app.py              # FastAPI application
├── Dockerfile          # Container build instructions
├── requirements.txt    # Python dependencies
├── .dockerignore
└── models/
    └── densenet121_chestxray.pth   # Model weights (download separately)
```

## Stopping the container

Press `Ctrl+C` in the terminal running the container, or if it's running in the background:

```bash
docker ps          # find the container ID
docker stop <container_id>
```

## Note

This is a research and educational tool, not a certified medical device. Predictions must be reviewed by a qualified radiologist and should not be used for actual clinical diagnosis.
