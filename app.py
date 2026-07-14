import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# Load Model
# ==========================================

MODEL_PATH = "best_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# Class Names
# ==========================================

class_names = [
    "Calculus",
    "Caries",
    "Gingivitis",
    "Hypodontia",
    "Mouth Ulcer",
    "Tooth Discoloration"
]

IMG_SIZE = (224, 224)

# ==========================================
# Prediction Function
# ==========================================

def predict(image):

    if image is None:
        empty_prediction = """
        <div class="empty-state">
            Upload an image and click Predict to see the result here.
        </div>
        """
        empty_probs = """
        <div class="empty-state">
            Class probabilities will appear here.
        </div>
        """
        return empty_prediction, empty_probs

    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    prediction = model.predict(arr, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    disease = class_names[predicted_index]
    confidence = float(prediction[predicted_index]) * 100

    prediction_html = f"""
    <div class="predicted-disease">{disease}</div>
    <div class="predicted-badge">✓ Predicted Disease</div>

    <div class="confidence-label">Confidence Score</div>
    <div class="confidence-value">{confidence:.2f}%</div>
    <div class="progress-track">
        <div class="progress-fill" style="width:{confidence:.2f}%;"></div>
    </div>

    <div class="info-note">
        🛡️ This prediction is based on the uploaded image analyzed by our deep learning model.
    </div>
    """

    rows = ""
    for i, name in enumerate(class_names):
        pct = float(prediction[i]) * 100
        is_top = "top" if i == predicted_index else ""
        rows += f"""
        <div class="prob-row">
            <div class="prob-name">{name}</div>
            <div class="prob-track">
                <div class="prob-fill {is_top}" style="width:{pct:.2f}%;"></div>
            </div>
            <div class="prob-value {is_top}">{pct:.2f}%</div>
        </div>
        """

    probabilities_html = f"""
    <div>{rows}</div>
    """

    return prediction_html, probabilities_html


# ==========================================
# Build Interface
# ==========================================

css = open("style.css", encoding="utf8").read()

with gr.Blocks(title="Oral Diseases AI", css=css, theme=gr.themes.Base()) as demo:

    gr.HTML("""
    <div class="hero-wrap">
        <div class="hero-title">
            <span class="tooth-icon">🦷</span>
            <span class="plain">Oral Diseases</span>
            <span class="gradient">Image Classification</span>
        </div>
        <p class="hero-subtitle">
            Upload an oral cavity image and the AI model will classify it
            into one of the six oral disease categories.
        </p>
        <div class="badge-pill">
            <span class="badge-icon">⊞</span>
            6 CLASSES &nbsp;•&nbsp; Calculus &nbsp;•&nbsp; Caries &nbsp;•&nbsp; Gingivitis
            &nbsp;•&nbsp; Hypodontia &nbsp;•&nbsp; Mouth Ulcer &nbsp;•&nbsp; Tooth Discoloration
        </div>
    </div>
    """)

    with gr.Row():

        with gr.Column(scale=1, elem_classes=["card-panel"]):
            gr.HTML("""
            <div class="card-heading"><span class="icon">☁️</span> Upload Oral Image</div>
            <div class="card-subtext">Drag &amp; drop an image here or click to browse</div>
            """)

            input_image = gr.Image(
                type="pil",
                label="",
                height=380,
                elem_classes=["upload-container"]
            )

            gr.HTML("""
            <div style="text-align:center; color:var(--blue-ice); opacity:0.6; font-size:12px; margin-top:8px;">
                Supports: JPG, JPEG, PNG &nbsp;•&nbsp; Max size: 10MB
            </div>
            """)

            predict_btn = gr.Button("🔍 Predict", elem_classes=["predict-btn"])

        with gr.Column(scale=1, elem_classes=["card-panel"]):
            gr.HTML("""
            <div class="card-heading"><span class="icon">✨</span> Prediction</div>
            """)

            prediction_output = gr.HTML("""
            <div class="empty-state">
                Upload an image and click Predict to see the result here.
            </div>
            """)

    with gr.Row(elem_classes=["card-panel"]):
        with gr.Column():
            gr.HTML("""
            <div class="card-heading">📊 Class Probabilities</div>
            <div class="card-subtext">The model's confidence for each class</div>
            """)

            probabilities_output = gr.HTML("""
            <div class="empty-state">
                Class probabilities will appear here.
            </div>
            """)

    gr.HTML("""
    <div class="footer-note">
        ℹ️ This AI model is for research and educational purposes only and should not be used
        as a substitute for professional medical advice.
    </div>
    """)

    predict_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[prediction_output, probabilities_output]
    )

    input_image.change(
        fn=predict,
        inputs=input_image,
        outputs=[prediction_output, probabilities_output]
    )

if __name__ == "__main__":
    demo.launch()
