# 🔞 NSFW Image Detection API Server

A FastAPI-powered backend service to detect NSFW (Not Safe For Work) content in images. This lightweight API enables developers to analyze images via a POST request and determine if they contain inappropriate or adult content.

---

## Features

- ⚡ FastAPI + Uvicorn backend
- Accepts image URLs or base64 image data
- Integrates with deep learning NSFW models (YOLOv8, CNNs, etc.)
- Auto-reloads during development using `--reload`
- Secure and production-ready with optional CORS, rate-limiting, and logging
- Docker-ready and deployable to cloud (AWS EC2, Heroku, Render, etc.)

---

## Project Structure

NSFW_Server/
├── api/
│ ├── init.py
│ ├── main.py # FastAPI app entry point
├── requirements.txt # Python dependencies
├── venv/ # Virtual environment 
└── README.md

---

## Setup

### Clone the Repository

```bash
git clone https://github.com/pyandcpp-coder/NSFW_Server.git
cd NSFW_Server
```

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Run the Development Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8300 --reload
```
##  API Usage

### Base URL
```bash
http://<your-server-ip>:8300
```

### POST /
```bash
{
  "image_url": "https://example.com/image.jpg"
}
```
