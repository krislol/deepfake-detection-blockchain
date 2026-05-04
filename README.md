# Deepfake Detection with Blockchain Verification

COMP1682 Final Year Project

This project detects deepfake videos using a fine-tuned deep learning model and verifies file authenticity using a blockchain. The two components work independently — the AI determines whether a video has been manipulated, while the blockchain confirms whether a file has been tampered with since it was registered.

# How It Works

A video is uploaded through the web interface. Frames are extracted and faces are detected using a Haar Cascade classifier. Each face crop is passed to an EfficientNet-B0 model which outputs a real/fake probability. These are averaged across all detected faces to produce a final video-level verdict.

Separately, a SHA-256 hash of the file can be registered on a simulated blockchain. Any future verification check compares the current file hash against the stored one to confirm the file is unmodified.

# Running

On Windows, double-click `run.bat`. This activates the environment, starts the FastAPI backend on port 8000, and opens the frontend in your browser. Install dependencies first with `pip install -r requirements.txt`.

# Model

The classifier uses EfficientNet-B0 pretrained on ImageNet, with a custom head added: two fully connected layers (1280→512→128→2) with ReLU activations and dropout. It was trained on Celeb-DF v2 using 150 real and 150 fake videos, achieving 98.7% validation accuracy and an AUC-ROC of 0.998.

