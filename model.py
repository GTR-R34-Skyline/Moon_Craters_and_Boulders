import tensorflow as tf
import os

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

def preprocess_image(image_path, target_size=(416, 416)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = image / 255.0  # Normalize to [0,1]
    return image

def load_image_and_labels(image_path, label_path):
    image = preprocess_image(image_path)
    
    # Load labels
    boxes, labels = tf.numpy_function(parse_yolo_label_file, [label_path], [tf.float32, tf.int32])
    boxes.set_shape([None, 4])  # Define shape for boxes
    labels.set_shape([None])    # Define shape for labels
    
    return image, (boxes, labels)

def load_dataset(image_dir, label_dir, target_size=(416, 416)):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    label_paths = [os.path.join(label_dir, fname.replace('.jpg', '.txt')) for fname in os.listdir(image_dir)]
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
    dataset = dataset.map(lambda image_path, label_path: load_image_and_labels(image_path, label_path))
    return dataset

# Example usage
dataset = load_dataset(r'C:\Users\Shashank\Desktop\isro_db\images', r'C:\Users\Shashank\Desktop\isro_db\final')
