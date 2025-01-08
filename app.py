from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Define the path for uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_image(image_path, target_size=(416, 416)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = image / 255.0  # Normalize to [0,1]
    return image

def predict_image(image_path, model):
    image = load_image(image_path)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return prediction

def parse_and_format_prediction(prediction):
    craters = []
    boulders = []

    if prediction.shape == (1, 4):
        for i, size in enumerate(prediction[0]):
            if i % 2 == 0:  # Assume even indices are craters and odd indices are boulders
                craters.append(size)
            else:
                boulders.append(size)
    else:
        print(f'Unexpected prediction format: {prediction}')

    return craters, boulders

def format_detection_results(craters, boulders):
    result = []
    result.append("Detection Results")
    result.append(f"Number of Craters Detected: {len(craters)}")
    result.append(f"Number of Boulders Detected: {len(boulders)}")
    result.append("")

    for i, size in enumerate(craters):
        result.append(f"Crater {i + 1}: Diameter {size:.2f} meters")
    
    for i, size in enumerate(boulders):
        result.append(f"Boulder {i + 1}: Size {size:.2f} meters")

    return "\n".join(result)

def visualize_prediction(image_path, craters, boulders):
    image = load_image(image_path)

    # Use plt.subplots to create a new figure and axis
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title('Detection Results')

    for i, size in enumerate(craters):
        ax.text(10, 20 + i * 20, f'Crater {i + 1}', color='blue')

    for i, size in enumerate(boulders):
        ax.text(10, 20 + (len(craters) + i) * 20, f'Boulder {i + 1}', color='red')

    ax.axis('off')

    buf = BytesIO()
    # Save the figure to the buffer
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)  # Close the figure to prevent memory leaks
    return image_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return redirect(request.url)

    # Save the first file
    file = files[0]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Load the model
    model_path = 'C:/Users/shash/Documents/College Projects/SPACE/isro_db/valid_model.keras'
    model = tf.keras.models.load_model(model_path)

    # Predict
    prediction = predict_image(file_path, model)
    craters, boulders = parse_and_format_prediction(prediction)
    formatted_output = format_detection_results(craters, boulders)
    image_base64 = visualize_prediction(file_path, craters, boulders)

    return render_template('results.html', results=formatted_output, image_data=image_base64)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
