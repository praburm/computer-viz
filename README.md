
# 🧠 AI Image Platform App – Kubernetes Deployment Guide

This guide helps you set up and deploy the AI Image Analysis platform using a single-node **K3s** Kubernetes cluster.

---

## 🚀 Prerequisites

- CV Dependencies
    - requirements.txt
- Docker
- K3s (single-node)
- Streamlit

---

## 📁 Project Structure

```
ai-image-analysis/
C:.
│   README.md
│
├───client
│   │   client_app.py
│   │
│   └───test-data
│       ├───coco
│       │       guitar.jpg
│       │       wrench.jpg
│       │       racket.jpg
│       │       istockphoto.jpg
│
├───patch-v2.0.0
│   │   app.py
│   │   client_app.py
│   │   yolov5s.pt
│   │
│   └───__pycache__
│           app.cpython-38.pyc
│
└───server
        app.py
        deployment.yaml
        Dockerfile
        port-forward.sh
        README.md
        requirements.txt
        service.yaml
```

---

## 🐳 Step 1: Build and Export Docker Image

```bash
# Build the Docker image
docker build -t image-platform-app .

# Save the image as a tar archive
docker save -o image-platform-app.tar image-platform-app
```

---

## 📦 Step 2: Import Docker Image into K3s

```bash
# Import image into K3s containerd
sudo k3s ctr images import image-platform-app.tar
```

---

## ☸️ Step 3: Deploy to K3s

```bash
# Install K3s (Single Node)
curl -sfL https://get.k3s.io | sh -

# Verify installation:
sudo kubectl get nodes

# Apply Kubernetes Deployment and Service YAMLs
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

---

## 🔌 Step 4: Port Forward the Service

Use the following shell script to safely port forward the K3s service.

```bash
bash port_forward.sh
```

---

## 🌐 Access the Application

Visit the app in your browser at:

```
http://localhost:30080
```

---

## 💻 Client Side: Image Analysis Platform App

This is the frontend Streamlit application that enables various image analysis functionalities:

### Features Provided

1. Image Detection  
2. Image Classification  
3. Image Segmentation  
4. Image Features Extraction  
5. Image Preprocessing  

### Setup Environment

```bash
# Create and activate Conda environment
conda create -n ai_img_torplat python=3.8 -y
conda activate ai_img_torplat

# Install dependencies
conda install -y -c pytorch pytorch torchvision torchaudio cudatoolkit=11.3
conda install -y -c conda-forge opencv pillow numpy matplotlib
pip install fastapi uvicorn nest-asyncio python-multipart ultralytics gradio
```

### Run Streamlit App

```bash
streamlit run client_app.py
```

### Test Images

Use the provided sample COCO images from:

```
test-data/coco/
test-data/analyze/
```

### 🔗 Note

Ensure the Streamlit app connects to the backend server via K3s on port `30080`.

---

## ✅ You're all set!
