# app.py
import os
from flask import Flask, render_template, request, redirect, flash, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Import dotenv to load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env into os.environ

# Import LangChain components
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

app = Flask(__name__)
app.secret_key = '123qazwsx'  

# Load your pre-trained model (.h5 file)
model = load_model('InceptionV3.h5')

# Retrieve the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in the environment. Check your .env file.")

def prepare_image(image_file, target_size=(224, 224)):
    """
    Load an image, resize it to the target size, convert it to an array,
    add a batch dimension, and normalize pixel values.
    """
    image = Image.open(image_file).convert("RGB")
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0  # Normalize if needed by your model
    return image_array

def generate_explanation(diagnosis):
    """
    Use LangChain with a custom prompt to generate a detailed explanation for the diagnosis.
    """
    custom_prompt = (
        "You are a knowledgeable medical assistant. Provide a detailed explanation for the eye condition below. "
        "Include information on possible causes, typical symptoms, and recommended treatments. "
        "Ensure that the explanation is clear and easy to understand for a layperson.\n\n"
        "Diagnosis: {diagnosis}"
    )
    
    # Create a prompt template with LangChain
    prompt_template = PromptTemplate(
        template=custom_prompt,
        input_variables=["diagnosis"]
    )
    
    # Initialize the OpenAI LLM via LangChain using the API key from the .env file
    llm = OpenAI(
        temperature=0.7,
        max_tokens=150,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Create a chain that combines the prompt with the LLM
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    try:
        explanation = chain.run(diagnosis=diagnosis)
    except Exception as e:
        explanation = "Error generating explanation. Please try again later."
        print("LangChain error:", e)
    return explanation

# Map numeric class indices to human-readable disease names.
CLASS_LABELS = {
    0:'Central Serous Retinopathy',
    1:'Diabetic Retinopathy',
    2:'Disc Edema',
    3:'Glaucoma',
    4:'Healthy',
    5:'Macular Scar',
    6:'Myopia',
    7:'Pterygium',
    8:'Retinal Detachment',
    9:'Retinitis Pigmentosa'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image is part of the request
        if 'image' not in request.files:
            flash('No image part in the request', 'danger')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        if file:
            try:
                # Preprocess the image and predict
                image = prepare_image(file)
                preds = model.predict(image)
                class_index = int(np.argmax(preds, axis=1)[0])
                diagnosis = CLASS_LABELS.get(class_index, "Unknown")
                # Generate an explanation using LangChain
                explanation = generate_explanation(diagnosis)
                # Render the same page with results
                return render_template('index.html', diagnosis=diagnosis, explanation=explanation)
            except Exception as e:
                flash(f"Error processing the image: {e}", 'danger')
                return redirect(request.url)
    # For GET requests or if no results are available, simply render the page without diagnosis
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)