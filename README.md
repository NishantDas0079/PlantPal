# 🌿 PlantPal – AI‑Powered Plant Disease Detector

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.onrender.com) <!-- Replace with your live Render URL -->
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**PlantPal** is an intelligent web application that helps plant lovers diagnose diseases from leaf photos. Using a deep learning model trained on the PlantVillage dataset, it identifies 15 common diseases across pepper, potato, and tomato plants with **93% validation accuracy**. The app provides confidence scores, detailed care instructions, and seasonal tips to keep your plants healthy.


## ✨ Features

- **📸 Instant Disease Detection** – Upload a leaf photo and get top‑3 disease predictions with confidence bars.
- **🌦️ Seasonal Care Tips** – Smart sidebar tips tailored to the current month (winter, spring, summer, autumn).
- **📊 Confidence Progress Bars** – Visual representation of prediction certainty.
- **📋 Detailed Care Cards** – For the top diagnosis, receive treatment, prevention, and notes.
- **💚 Daily Plant Tips** – Random helpful advice in the sidebar.
- **🎨 Polished UI** – Clean, plant‑themed design with smooth interactions.

## 🛠️ Tech Stack

| Component       | Technology                         |
|-----------------|------------------------------------|
| Frontend        | [Streamlit](https://streamlit.io/) |
| Backend/ML      | [PyTorch](https://pytorch.org/)    |
| Model           | ResNet‑18 (transfer learning)      |
| Dataset         | PlantVillage (15‑class subset)     |
| Deployment      | [Render](https://render.com/)      |

## 🚀 Live Demo

Check out the live app: [PlantPal on Render](https://plantpal-jg5r.onrender.com) <!-- Replace with your URL -->


### Prerequisites

- Python 3.8 or higher
- Git

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/PlantPal.git
   cd PlantPal
   ```

2. **Create and activate virtual environment**
   ```
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```
   streamlit run src/app.py
   ```

Open your browser and go to `http://localhost:8501`

# Model Training

The model was trained on a subset of the `PlantVillage` dataset containing 20,638 images across 15 classes (pepper, potato, and tomato diseases). A pre‑trained `ResNet‑18` was fine‑tuned for 10 epochs with data augmentation. Final validation accuracy reached `93%`.

The trained model `(plant_disease_resnet18.pth)` and class names `(class_names.pkl)` are included in the `models/ folder`.

# ☁️ Deployment on Render
PlantPal is deployed on Render using a free web service. The deployment configuration:

Build Command: `pip install -r requirements.txt`

Start Command: `streamlit run src/app.py --server.port $PORT --server.address 0.0.0.0`

The live app may take a few seconds to wake up after inactivity (free tier).

# License

This project is open source and available under the `Apache 2.0`

# 🛠️ Future Enhancements
Expand dataset to all 38 PlantVillage classes

Add camera capture functionality

Include more detailed treatment databases

Implement user accounts to save plant history
