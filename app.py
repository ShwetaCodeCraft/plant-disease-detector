import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model and class indices
model = tf.keras.models.load_model("plant_disease_model.keras")
class_indices = json.load(open("class_indices.json"))

def predict(image):
    if image is None:
        return "Please upload an image!", "", ""
    
    # Preprocess
    img = Image.fromarray(image).resize((224, 224))
    img_array = np.expand_dims(np.array(img).astype('float32') / 255., axis=0)
    
    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_indices[str(np.argmax(prediction))]
    confidence = float(np.max(prediction)) * 100
    
    # Parse result
    parts = predicted_class.split('___')
    plant = parts[0].replace('_', ' ')
    disease = parts[1].replace('_', ' ') if len(parts) > 1 else predicted_class
    
    is_healthy = 'healthy' in disease.lower()
    status = "✅ Healthy Plant" if is_healthy else "⚠️ Disease Detected"
    
    return (
        f"{disease}",
        f"🌿 Plant: {plant}",
        f"{status} — Confidence: {confidence:.2f}%"
    )

# Custom CSS
css = """
body { font-family: 'Segoe UI', sans-serif; }
.gradio-container { 
    background: linear-gradient(135deg, #1a3a2a 0%, #2d6a4f 100%) !important;
    min-height: 100vh;
}
.main-header {
    text-align: center;
    padding: 30px 0 10px;
    color: #b7e4c7;
}
.main-header h1 {
    font-size: 3em;
    font-weight: 900;
    color: #52b788;
    margin-bottom: 8px;
}
.main-header p {
    color: #95d5b2;
    font-size: 1.1em;
}
footer { display: none !important; }
"""

with gr.Blocks(css=css, title="PlantScan - Disease Detector") as demo:
    gr.HTML("""
        <div class="main-header">
            <h1>🌿 PlantScan</h1>
            <p>AI-powered plant disease detection using CNN Deep Learning</p>
            <p style="font-size:0.85em; color:#74c69d; margin-top:6px">
                Trained on 54,000+ images • 38 disease classes • 97% accuracy
            </p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Leaf Image",
                type="numpy",
                height=300
            )
            detect_btn = gr.Button(
                "🔬 Analyse Leaf",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):
            disease_output = gr.Textbox(
                label="🦠 Disease Detected",
                interactive=False,
                lines=2
            )
            plant_output = gr.Textbox(
                label="🌱 Plant",
                interactive=False
            )
            status_output = gr.Textbox(
                label="📊 Status & Confidence",
                interactive=False
            )

    gr.HTML("""
        <div style="text-align:center; margin-top:20px; color:#74c69d; font-size:0.85em">
            <p>Supported plants: Apple, Tomato, Potato, Corn, Grape, Peach, Strawberry & more</p>
            <p style="margin-top:6px">Built with TensorFlow • FastAPI • Gradio | By Shweta Singh</p>
        </div>
    """)

    detect_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[disease_output, plant_output, status_output]
    )

    gr.Examples(
        examples=[],
        inputs=image_input
    )

demo.launch()
