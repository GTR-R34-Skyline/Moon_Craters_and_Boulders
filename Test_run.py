import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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

# Path to a new image for prediction
new_image_path = r'C:\Users\shash\Documents\College Projects\SPACE\isro_db\valid\10_png.rf.bebc80626ea00941670c0e220e275261.jpg'

# Load the saved model
model = tf.keras.models.load_model(r'C:\Users\shash\Documents\College Projects\SPACE\isro_db\valid_model.keras')

# Make a prediction
prediction = predict_image(new_image_path, model)
print('Prediction:', prediction)
print('Prediction shape:', prediction.shape)
print('Prediction type:', type(prediction))

# Function to parse and format prediction
def parse_and_format_prediction(prediction):
    craters = []
    boulders = []

    # Debugging: Print the prediction structure
    print(f'Full prediction: {prediction}')

    # Updated parsing logic based on the new prediction format
    if prediction.shape == (1, 4):
        for i, size in enumerate(prediction[0]):
            print(f'Prediction element {i}: {size}')
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

# Parse and format prediction results
craters, boulders = parse_and_format_prediction(prediction)
formatted_output = format_detection_results(craters, boulders)
print(formatted_output)

# Visualize the result
def visualize_prediction(image_path, craters, boulders):
    image = load_image(image_path)
    plt.imshow(image)
    plt.title('Detection Results')

    # Plot craters
    for i, size in enumerate(craters):
        plt.text(10, 20 + i * 20, f'Crater {i + 1} ', color='blue')

    # Plot boulders
    for i, size in enumerate(boulders):
        plt.text(10, 20 + (len(craters) + i) * 20, f'Boulder {i + 1}', color='red')

    plt.axis('off')
    plt.show()

visualize_prediction(new_image_path, craters, boulders)
