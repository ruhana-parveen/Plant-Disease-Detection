# Plant Disease Detection using CNN

A deep-learning image-classification mini project built with TensorFlow/Keras and Gradio. It classifies plant leaf images into the 38 PlantVillage categories.

## Features
- CNN image classifier trained on PlantVillage
- 128 × 128 RGB image preprocessing
- Top-3 predicted classes with confidence scores
- Gradio web interface
- Training and validation accuracy/loss plots
- Confusion matrix and classification report

> **Important:** This is an educational demonstration, not a substitute for diagnosis by an agricultural expert. Predictions can be unreliable for images outside the PlantVillage dataset.

## Project structure

```text
Plant_Disease_Detection_CNN/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
├── models/
│   ├── plant_disease_cnn.keras   # Add your trained model here
│   └── class_names.json          # Export from your Colab notebook
├── screenshots/
│   └── Add your own app/output screenshots here
└── reports/
    └── project_report.md
```

## 1. Get the trained model and class labels from Colab

Your trained model and the exact class order must match. In your existing Colab notebook, run:

```python
import json
from google.colab import files

# Save the exact label order used while training
class_names = info.features["label"].names

with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

# Save model if you have not already saved it
model.save("plant_disease_cnn.keras")

files.download("plant_disease_cnn.keras")
files.download("class_names.json")
```

Put both downloaded files inside the repository's `models/` folder:
- `models/plant_disease_cnn.keras`
- `models/class_names.json`

Do not create or reorder the class names manually.

## 2. Install dependencies

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## 3. Run the web app

From the project root:

```bash
python app.py
```

Open the local URL printed by Gradio, usually `http://127.0.0.1:7860`.

## 4. Train from scratch (optional)

The `train.py` script downloads PlantVillage through TensorFlow Datasets, trains the same CNN architecture, and saves the model and class names in `models/`.

```bash
python train.py
```

Training can take time and requires a stable internet connection and sufficient RAM/GPU memory.

## 5. Push to GitHub

Create a new empty repository on GitHub, then run these commands from this folder:

```bash
git init
git add .
git commit -m "Add Plant Disease Detection CNN project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your own repository URL.

## Evaluation notes

- Dataset: PlantVillage via TensorFlow Datasets (`plant_village`)
- Image size: 128 × 128
- Model: 3 convolution + max-pooling blocks, dense classifier, dropout, softmax output
- Output: one of 38 PlantVillage classes
- Report your actual training/test metrics from your own run. Do not claim a test accuracy unless you evaluated on a separate held-out test set.
