import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import pickle
import numpy as np
import random
from datetime import datetime

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="🌿 PlantPal",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------- Custom CSS -------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #d4edda 0%, #a3d9a5 100%);
        font-family: 'Poppins', sans-serif;
    }

    h1, h2, h3 {
        color: #1b5e20;
        font-weight: 600;
    }

    .stFileUploader {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 2px dashed #66bb6a;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }

    /* Prediction cards */
    .pred-card {
        background: white;
        border-radius: 20px;
        padding: 15px 25px;
        margin: 15px 0;
        border-left: 8px solid #4caf50;
        box-shadow: 0 8px 20px rgba(0,100,0,0.1);
        transition: transform 0.2s ease;
    }
    .pred-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0,100,0,0.15);
    }
    .pred-card h4 {
        margin: 0 0 10px 0;
        color: #1e3a2f;
        font-weight: 600;
    }

    /* Progress bar inside card */
    .progress-container {
        width: 100%;
        background-color: #e0e0e0;
        border-radius: 25px;
        margin: 8px 0;
        height: 20px;
        overflow: hidden;
    }
    .progress-bar {
        height: 20px;
        border-radius: 25px;
        background: linear-gradient(90deg, #66bb6a, #43a047);
        text-align: right;
        padding-right: 8px;
        color: white;
        font-size: 0.8rem;
        line-height: 20px;
        font-weight: 600;
    }

    /* Expander (treatment) */
    .streamlit-expanderHeader {
        background-color: #e8f5e9;
        border-radius: 12px;
        font-weight: 600;
        color: #1b5e20;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #81c784 0%, #66bb6a 100%);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #66bb6a, #43a047);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 10px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #43a047, #2e7d32);
        transform: scale(1.02);
    }

    .stAlert {
        border-radius: 15px;
        border-left: 5px solid #4caf50;
    }

    .stSpinner > div {
        border-top-color: #4caf50 !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------- Load Model & Classes -------------------
@st.cache_resource
def load_model():
    with open("models/class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("models/plant_disease_resnet18.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model, class_names, device

model, class_names, device = load_model()

# ------------------- Image Transform -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------- Detailed Care Database -------------------
# You can expand this dictionary with more diseases as needed.
# The keys should exactly match the class names from your dataset.
care_db = {
    "Pepper__bell___Bacterial_spot": {
        "treatment": "Copper-based fungicides, remove infected leaves, avoid overhead watering.",
        "prevention": "Use disease-free seeds, crop rotation, resistant varieties.",
        "notes": "Bacterial spot spreads rapidly in warm, wet conditions."
    },
    "Pepper__bell___healthy": {
        "treatment": "No treatment needed – your plant is healthy!",
        "prevention": "Maintain good watering practices and monitor regularly.",
        "notes": "Keep up the great care!"
    },
    "Potato___Early_blight": {
        "treatment": "Apply fungicides (chlorothalonil), prune affected leaves, mulch to prevent soil splash.",
        "prevention": "Rotate crops, water at base, use disease-free seed potatoes.",
        "notes": "Early blight appears as dark spots with concentric rings."
    },
    "Potato___Late_blight": {
        "treatment": "Destroy infected plants immediately, apply fungicides (copper-based), avoid wet foliage.",
        "prevention": "Use resistant varieties, ensure good air circulation, plant in well-drained soil.",
        "notes": "Late blight caused the Irish Potato Famine – act fast!"
    },
    "Potato___healthy": {
        "treatment": "None needed.",
        "prevention": "Continue regular care.",
        "notes": "Your potato plant is healthy."
    },
    "Tomato_Bacterial_spot": {
        "treatment": "Copper sprays, remove symptomatic leaves, avoid working with wet plants.",
        "prevention": "Use disease-free seeds, rotate crops, stake plants for airflow.",
        "notes": "Bacterial spot causes small water-soaked spots on leaves and fruit."
    },
    "Tomato_Early_blight": {
        "treatment": "Fungicides (chlorothalonil), mulch, prune lower leaves.",
        "prevention": "Water at base, rotate crops, use resistant cultivars.",
        "notes": "Early blight starts on lower leaves with target-like spots."
    },
    "Tomato_Late_blight": {
        "treatment": "Remove infected plants, apply copper fungicides, avoid overhead irrigation.",
        "prevention": "Plant resistant varieties, space plants for airflow, monitor humidity.",
        "notes": "Late blight causes dark, greasy-looking lesions on leaves and stems."
    },
    "Tomato_Leaf_Mold": {
        "treatment": "Improve ventilation, reduce humidity, apply fungicides if severe.",
        "prevention": "Avoid overcrowding, water in morning, use resistant varieties.",
        "notes": "Leaf mold causes yellow spots on upper leaf surface and olive-green mold underneath."
    },
    "Tomato_Septoria_leaf_spot": {
        "treatment": "Remove infected leaves, apply fungicides, mulch to prevent soil splash.",
        "prevention": "Crop rotation, stake plants, water at base.",
        "notes": "Septoria appears as small circular spots with dark borders."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "treatment": "Insecticidal soap, neem oil, increase humidity, prune heavily infested leaves.",
        "prevention": "Regularly mist plants, inspect undersides of leaves.",
        "notes": "Spider mites cause stippling and fine webbing."
    },
    "Tomato__Target_Spot": {
        "treatment": "Apply fungicides, remove affected foliage, avoid wet leaves.",
        "prevention": "Use resistant varieties, ensure good air circulation.",
        "notes": "Target spot causes brown lesions with concentric rings."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "treatment": "Remove infected plants to prevent spread, control whiteflies (insecticidal soap).",
        "prevention": "Use reflective mulches, plant resistant varieties, exclude whiteflies with row covers.",
        "notes": "Yellow leaf curl virus causes stunted growth and curling leaves."
    },
    "Tomato__Tomato_mosaic_virus": {
        "treatment": "No cure – remove and destroy infected plants, wash hands and tools.",
        "prevention": "Use resistant varieties, wash hands before handling, disinfect tools.",
        "notes": "Mosaic virus causes mottled leaves and distorted growth."
    },
    "Tomato_healthy": {
        "treatment": "None.",
        "prevention": "Continue regular care.",
        "notes": "Your tomato plant is healthy!"
    }
}

# ------------------- Seasonal Tips -------------------
def get_seasonal_tip():
    month = datetime.now().month
    if month in [12, 1, 2]:  # Winter
        tips = [
            "❄️ Reduce watering in winter – plants need less water.",
            "❄️ Keep plants away from cold drafts and windows.",
            "❄️ Avoid fertilizing during dormancy."
        ]
    elif month in [3, 4, 5]:  # Spring
        tips = [
            "🌸 Spring is growth time! Start fertilizing lightly.",
            "🌸 Repot root‑bound plants now.",
            "🌸 Increase watering as days get longer."
        ]
    elif month in [6, 7, 8]:  # Summer
        tips = [
            "☀️ Water early morning to prevent evaporation.",
            "☀️ Provide shade for sensitive plants during peak sun.",
            "☀️ Watch for pests – they love summer too!"
        ]
    else:  # Autumn
        tips = [
            "🍂 Reduce watering as growth slows.",
            "🍂 Bring outdoor plants inside before first frost.",
            "🍂 Clean up fallen leaves to prevent disease."
        ]
    return random.choice(tips)

# ------------------- Sidebar Content -------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/plant-under-sun.png", width=100)
    st.markdown("## 🌱 PlantPal")
    st.markdown("Your intelligent plant care companion")

    st.markdown("---")
    st.markdown("### 🌦️ Seasonal Tip")
    st.info(get_seasonal_tip())

    st.markdown("---")
    st.markdown("### 💚 Daily Care Tip")
    daily_tips = [
        "💧 Water in the morning to reduce evaporation.",
        "☀️ Most houseplants need bright, indirect light.",
        "🍂 Yellow leaves? You might be overwatering.",
        "🌿 Wipe dusty leaves to help them breathe better.",
        "🐞 Check for pests weekly – early detection is key!",
        "🌸 Rotate your plants regularly for even growth.",
        "🧪 Use a balanced fertilizer during growing season.",
        "💨 Keep plants away from drafts and air conditioners."
    ]
    st.info(random.choice(daily_tips))

    st.markdown("---")
    st.caption("Built with PyTorch & Streamlit")

# ------------------- Main UI -------------------
st.title("🌿 PlantPal – Plant Disease Detector")
st.markdown("Upload a photo of a leaf, and I'll identify possible diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Your Leaf", use_container_width=True)

    with st.spinner("🌱 Analyzing..."):
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            top3_idx = torch.topk(probs, min(3, len(class_names))).indices.cpu().numpy()
            top3_probs = probs[top3_idx].cpu().numpy()

    # Show results with progress bars
    st.subheader("🔍 Diagnosis Results")
    for i, idx in enumerate(top3_idx):
        disease = class_names[idx]
        confidence = top3_probs[i]
        if "healthy" in disease.lower():
            emoji = "✅"
        else:
            emoji = "⚠️"

        # Progress bar as percentage
        percent = int(confidence * 100)
        st.markdown(f"""
        <div class="pred-card">
            <h4>{emoji} {i+1}. {disease}</h4>
            <div class="progress-container">
                <div class="progress-bar" style="width:{percent}%;">{percent}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Detailed care for the top prediction
    top_disease = class_names[top3_idx[0]]
    if top_disease in care_db:
        with st.expander(f"🌿 Detailed Care for {top_disease}"):
            care = care_db[top_disease]
            st.markdown(f"**Treatment:** {care['treatment']}")
            st.markdown(f"**Prevention:** {care['prevention']}")
            st.markdown(f"**Notes:** {care['notes']}")
    else:
        with st.expander(f"🌿 Detailed Care for {top_disease}"):
            st.markdown("Detailed care information not available for this disease yet.")
            st.markdown("General tip: Remove affected leaves and improve air circulation.")

    # Quick treatment suggestion (if diseased)
    if not any("healthy" in class_names[i].lower() for i in top3_idx):
        with st.expander("💊 Quick Treatment Summary"):
            st.markdown("""
            - ✂️ Remove affected leaves immediately
            - 💨 Improve air circulation
            - 🌿 Apply organic fungicide if fungal
            - 💧 Water at base, not overhead
            """)
    else:
        st.success("✅ Your plant looks healthy! Keep up the good care.")

# ------------------- Footer -------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #4caf50;'>🌍 Made with 🌱 for plant lovers everywhere</p>",
    unsafe_allow_html=True
)