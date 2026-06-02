================================================================
       PLANT DISEASE DETECTOR - PROJECT README
       By: Shweta Singh | RVS College of Engineering
================================================================

ABOUT THE PROJECT
-----------------
This project uses a CNN (Convolutional Neural Network) deep learning
model to detect plant diseases from leaf images. The model was trained
on the PlantVillage dataset with 54,000+ images across 38 disease classes
and achieves 97% training accuracy.

Tech Stack:
- Python 3.11
- TensorFlow / Keras (CNN Model)
- FastAPI (Backend)
- HTML / CSS / JavaScript (Frontend)

----------------------------------------------------------------

FOLDER STRUCTURE
----------------
PlantDiseaseApp/
├── main.py                                  → FastAPI backend
├── class_indices.json                       → 38 disease class labels
├── plant_disease_model.keras                → Trained CNN model (573MB)
├── Plant_Disease_Prediction_CNN_Image_Classifier.ipynb  → Training notebook
├── README.txt                               → This file
└── templates/
    └── index.html                           → Frontend UI

----------------------------------------------------------------

HOW TO RUN THE PROJECT
----------------------

STEP 1 — Install Python 3.11
  Download from: https://www.python.org/downloads/release/python-3119/
  ⚠️ IMPORTANT: Check "Add Python to PATH" during installation!

STEP 2 — Open CMD/Terminal inside the PlantDiseaseApp folder
  - Right click inside the folder
  - Click "Open in Terminal" or "Open Command Prompt here"

STEP 3 — Install required packages (only once)
  Run this command:
  py -3.11 -m pip install fastapi uvicorn tensorflow pillow python-multipart jinja2

  (This will take 3-5 minutes to download)

STEP 4 — Start the server
  Run this command:
  py -3.11 -m uvicorn main:app --reload

  You should see:
  INFO: Uvicorn running on http://127.0.0.1:8000

STEP 5 — Open the app in browser
  Go to: http://127.0.0.1:8000

----------------------------------------------------------------

HOW TO USE THE APP
------------------
1. Click "Choose File" or drag and drop a leaf image
2. Click "Analyse Leaf" button
3. The app will show:
   - Disease name
   - Plant name
   - Confidence score (%)
   - Whether plant is Healthy or Diseased

----------------------------------------------------------------

SUPPORTED PLANTS & DISEASES (38 Classes)
-----------------------------------------
Apple, Blueberry, Cherry, Corn, Grape, Orange,
Peach, Pepper, Potato, Raspberry, Soybean,
Squash, Strawberry, Tomato

----------------------------------------------------------------

MODEL DETAILS
-------------
- Architecture  : CNN (Convolutional Neural Network)
- Dataset       : PlantVillage (Kaggle)
- Total Images  : 54,305 (43,456 train / 10,849 validation)
- Classes       : 38 (diseases + healthy)
- Image Size    : 224 x 224 pixels
- Epochs        : 5
- Train Accuracy: 97.8%
- Val Accuracy  : 87.0%
- Framework     : TensorFlow / Keras

----------------------------------------------------------------

NOTE
----
⚠️ Make sure plant_disease_model.keras (573MB) is present
   in the folder. Without it the app will NOT start.

⚠️ No internet connection required to run the app.
   Everything runs 100% locally on your laptop.

================================================================
