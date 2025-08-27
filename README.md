# 🚀 FastAPI Stable Diffusion API

A simple API for generating images with **Stable Diffusion**, built using [FastAPI](https://fastapi.tiangolo.com/).  
Supports LoRA models, background email notifications, and comes with interactive API docs.

---

## ✨ Features
- Generate images from text prompts  
- Plug-and-play **LoRA support** (`.safetensors` files)  
- Background **email notifications**  
- Built-in interactive docs at `/docs`  

---

## 🔧 Setup

### Requirements
- Python **3.11**
- NVIDIA GPU (≥ 8GB VRAM recommended)

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
---
#### Run 
🚀 Run

Start the API with Uvicorn
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Then open your browser at:
Swagger Docs: http://127.0.0.1:8000/docs

---

📚 Example Usage
Generate an Image
```bash
curl -X POST "http://127.0.0.1:8000/generate-image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A futuristic cityscape at night in cyberpunk style",
    "lora_name": "cyberpunk-v1",
    "lora_scale": 0.8
  }'
```
