import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Parse YOLO label files
def parse_yolo_label_file(label_file):
    boxes = []
    labels = []
    with tf.io.gfile.GFile(label_file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            parts = line.split()
            labels.append(tf.cast(int(parts[0]), tf.int32))
            x_center, y_center, width, height = map(float, parts[1:])
            boxes.append([tf.cast(x_center, tf.float32), tf.cast(y_center, tf.float32),
                          tf.cast(width, tf.float32), tf.cast(height, tf.float32)])
    return tf.convert_to_tensor(boxes, dtype=tf.float32), tf.convert_to_tensor(labels, dtype=tf.int32)

# Convert to TensorFlow-compatible types
def parse_yolo_label_file_tf(label_file):
    boxes, labels = tf.numpy_function(parse_yolo_label_file, [label_file], [tf.float32, tf.int32])
    boxes.set_shape([None, 4])  # Define shape for boxes
    labels.set_shape([None])    # Define shape for labels
    return boxes, labels

# Preprocess images
def preprocess_image(image_path, target_size=(416, 416)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = image / 255.0  # Normalize to [0,1]
    return image

# Load image and labels
def load_image_and_labels(image_path, label_path):
    image = preprocess_image(image_path)
    boxes, labels = parse_yolo_label_file_tf(label_path)
    return image, (boxes, labels)

# Load dataset
def load_dataset(image_dir, label_dir, target_size=(416, 416)):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    label_paths = [os.path.join(label_dir, fname.replace('.jpg', '.txt')) for fname in os.listdir(image_dir)]
    valid_image_paths = []
    valid_label_paths = []

    for img_path, lbl_path in zip(image_paths, label_paths):
        if os.path.exists(lbl_path):
            valid_image_paths.append(img_path)
            valid_label_paths.append(lbl_path)

    dataset = tf.data.Dataset.from_tensor_slices((valid_image_paths, valid_label_paths))
    dataset = dataset.map(lambda image_path, label_path: load_image_and_labels(image_path, label_path))
    return dataset

# Prepare dataset
def prepare_data(dataset, batch_size=32, buffer_size=100):
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Define paths
image_dir = r'C:\Users\Shashank\Desktop\isro_db\images'
label_dir = r'C:\Users\Shashank\Desktop\isro_db\final'

dataset = load_dataset(image_dir, label_dir)
dataset = prepare_data(dataset)

# Define and create model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(416, 416, 3)),  # Updated to use 'shape' instead of 'input_shape'
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4)  # Adjust output units based on your specific needs
    ])
    model.compile(optimizer='adam', loss='mse')  # Use appropriate loss function
    return model

model = create_model()

# Train the model
history = model.fit(dataset, epochs=10)  # Adjust epochs and other parameters as needed

# Evaluate the model
evaluation = model.evaluate(dataset)
print('Evaluation results:', evaluation)

# Save the model with .keras extension
model.save(r'C:\Users\Shashank\Desktop\isro_db\valid_model.keras')

