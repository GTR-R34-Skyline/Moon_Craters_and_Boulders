import tensorflow as tf
import os
import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Parse YOLO label files
def parse_yolo_label_file(label_file):
    boxes = []
    labels = []
    with tf.io.gfile.GFile(label_file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            parts = line.split()
            labels.append(int(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:])
            boxes.append([x_center, y_center, width, height])
    return boxes, labels

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
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
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
        tf.keras.layers.InputLayer(input_shape=(416, 416, 3)),
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

# Save the model
model.save(r'C:\Users\Shashank\Desktop\isro_db\valid')
