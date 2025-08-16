# Project Rules & Context — Medical Image Classifier

## 1. Project Overview
This project is a **CNN-based image classification system** for **medical imaging** (e.g., pneumonia detection in chest X-rays).  
We use **transfer learning** with a pretrained CNN (ResNet50, EfficientNet, etc.) to speed up training and improve accuracy on limited medical data.

---

## 2. Architecture
**Root Folder Structure:**

medical_image_classifier/
│
├── data/                       # Dataset (local copies, ignored by Git)
│   ├── raw/                    # Original Kaggle dataset (unchanged)
│   ├── processed/              # Preprocessed images ready for training (none)
│
├── notebooks/                  # Jupyter notebooks for exploration
│   └── 01_data_check.ipynb       # Model training and experiments
│
├── src/                        # All Python modules
│   ├── __init__.py
│   ├── config.py               # All constants and configs
│   ├── data_loader.py          # Dataset loading & preprocessing
│   ├── model.py                # CNN architecture + transfer learning
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation metrics
|   ├── logger.py               # Track the application's behavior
│
├── app/                        # FastAPI backend
│   ├── __init__.py
│   ├── main.py                 # API routes
│   ├── predict.py               # Load model + inference logic
│
├── ui/                         # Gradio or Streamlit interface
│   └── interface.py
│
├── tests/                      # Unit tests
│   └── test_predict.py
│
├── .env                        # Environment variables (ignored by Git)
├── .env.example                # example environment variables if needed
├── .gitignore
├── .dockerignore
├── requirements.txt
├── README.md
└── dockerfile

---

## 3. Naming Conventions
- **Folders**: lowercase with underscores (e.g., `data_loader.py`).
- **Python variables**: snake_case (e.g., `image_size = 224`).
- **Classes**: PascalCase (e.g., `ImageClassifier`).
- **Constants**: UPPERCASE (e.g., `BATCH_SIZE = 32`).
- **Function names**: snake_case (e.g., `train_model()`).
- **Environment variables**: Uppercase with underscores (e.g., `MODEL_PATH`).

---

## 4. Technologies Used
- **Python 3.11.9+**
- **PyTorch** — Deep learning framework for model building.
- **torchvision** — Pretrained models & transforms.
- **FastAPI** — Backend API to serve predictions.
- **Gradio** — Web UI for easy testing.
- **Pillow (PIL)** — Image processing.
- **scikit-learn** — Metrics and evaluation.
- **matplotlib / seaborn** — Data visualization.
- **Hugging Face Spaces** — Deployment.
- **Docker** — Containerization.
- **Git/GitHub** — Version control.

---

## 5. Workflow
1. **Data Collection & Preprocessing**
2. **Model Definition**
3. **Training**
4. **Evaluation**
5. **Serving Predictions (API + UI)**
6. **Deployment**

---

## 6. Design Principles
- **Modular Code** — Separate concerns into distinct files (data, model, training, API).
- **PEP8 Compliance** — Follow Python style guide.
- **Reusability** — Functions should be flexible for future datasets.
- **Documentation** — Clear docstrings, comments, and README updates.
- **Testability** — Every major function should have at least one test.
- **Scalability** — Design so it can handle larger datasets in the future.

---

## 7. Future Agent Instructions
If an AI agent is assisting:
- Do NOT change architecture without permission.
- Always keep `.env` files out of version control.
- Maintain naming conventions and docstring style.
- If unsure about a design decision, ask before implementing.
- When modifying code, preserve existing functionality unless instructed otherwise.
- Always use windows cmd commands 
