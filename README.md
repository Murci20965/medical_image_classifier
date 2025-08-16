# 🩺 Pneumonia Detection from Chest X-Ray Images

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Ready-brightgreen?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **end-to-end**, production-grade deep learning application that classifies **chest X-ray images** as either **Normal** or **Pneumonia**.  
Built with **PyTorch**, served via **FastAPI**, and accessible through an interactive **Gradio** UI — all fully **Dockerized**.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Testing](#testing)
- [Deployment](#deployment)
- [License](#license)

---

## 📖 Overview
This project uses **transfer learning** with a pre-trained **ResNet50** model for accurate pneumonia detection from chest X-rays.  
The architecture cleanly separates core ML logic, the API backend, and the UI frontend, ensuring scalability, maintainability, and deployment flexibility.

---

## ✨ Features
✅ **High-Performance Model** – PyTorch CNN + Transfer Learning  
✅ **RESTful API** – Built with FastAPI for scalable inference  
✅ **Interactive UI** – Drag & drop image upload with Gradio  
✅ **Dockerized** – One command deploy anywhere  
✅ **MLOps-Ready** – Versioned models, automated tests, clean workflow

---

## 📊 Model Performance
The model was trained for 10 epochs, achieving a final validation accuracy of **87.50%**. On the unseen test set, the model's performance was evaluated as follows:

- **Overall Accuracy**: **82.85%**
- **Key Metrics for Pneumonia Class**:
  - **Precision**: 0.91 (Of all the images predicted as "Pneumonia", 91% were correct.)
  - **Recall**: 0.96 (The model correctly identified 96% of all actual "Pneumonia" cases.)
  - **F1-Score**: 0.88 (A balanced measure of precision and recall.)

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| NORMAL    | 0.91      | 0.60   | 0.72     | 234     |
| PNEUMONIA | 0.80      | 0.96   | 0.88     | 390     |
| **Total** | **0.84** | **0.83** | **0.82** | **624** |


---

## 📂 Dataset
**Source:** [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)  
**Size:** 5,863 images (JPEG) — split into `Pneumonia` and `Normal`.  
**Origin:** Pediatric patients (1–5 years old) from Guangzhou Women and Children’s Medical Center, China.  

---

## 🛠 Tech Stack
- **Backend:** Python, FastAPI  
- **Deep Learning:** PyTorch, Torchvision  
- **UI:** Gradio  
- **Containerization:** Docker  
- **Testing:** Pytest  
- **Utilities:** Scikit-learn, Pillow, Matplotlib, Seaborn  

---

## Project Architecture
The project follows a modular structure to ensure a clean separation of concerns:
```
medical_image_classifier/
│
├── app/              # FastAPI backend source code
├── data/             # (Local) Dataset storage (ignored by Git) (create folder manually, download data from kaggle using the link, unzip inside data/raw/)
├── logs/             # Stores application log files (auto generated during train and evaluation)
├── models/           # Trained and versioned model files (folder auto generated during training if not exist)
├── src/              # Core source code (data loading, model, training, etc.)
├── tests/            # Pytest unit tests
├── ui/               # Gradio user interface source code
├── .dockerignore     # Specifies files to ignore in the Docker build
├── .gitignore        # Specifies files to ignore for Git
├── Dockerfile        # Recipe for building the application container
├── pytest.ini        # Pytest configuration
├── requirements.txt  # Project dependencies
└── README.md         # Project documentation
```
---

## Setup and Installation

Follow these steps to set up the project on your local machine.

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/Murci20965/medical_image_classifier.git](https://github.com/Murci20965/medical_image_classifier.git)
   cd medical_image_classifier
   ```

2. **Create and activate a virtual environment:**

   ```bash
   # For Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset:**
   Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place the contents into the `data/raw/` directory. The final path should be `data/raw/chest_xray/`.

---
## Usage Guide

Ensure your virtual environment is active before running any commands.

### 1. Training the Model

To train the model from scratch, run the following command from the project's root directory. This will save a new, timestamped model file in the `models/` directory.

```bash
python -m src.train
```

### 2. Evaluating the Model

To evaluate the performance of the **latest** trained model on the unseen test set, run:

```bash
python -m src.evaluate
```

This will print a classification report to the console and save a confusion matrix plot to the `plots/` folder.

### 3. Running the Application (Local)

To run the application locally, you will need two separate terminals.

1. **Terminal 1: Start the API Server**

   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

   The API documentation will be available at `http://12-7.0.0.1:8000/docs`.

2. **Terminal 2: Start the Gradio UI**

   ```bash
   python ui/interface.py
   ```

   Open the local URL provided in the terminal to access the web interface.

### 4. Running with Docker

Ensure Docker Desktop is installed and running.

1. **Build the Docker image:**

   ```bash
   docker build -t medical-image-classifier .
   ```

2. **Run the container:**

   ```bash
   docker run --rm -p 8000:8000 --name medical-app medical-image-classifier
   ```

   The API will be running inside the container and accessible at `http://localhost:8000`. You can then start the Gradio UI locally (Step 3B) to interact with it.

---
## Testing

To run the automated unit tests and ensure the application's core logic is working correctly, use the following command:

```bash
python -m pytest
```

---
## Deployment

This application is fully containerized and ready for deployment. The `Dockerfile` can be used to deploy the application to any cloud service that supports containers, such as AWS, Azure, or Google Cloud Platform.

---
## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
