import os
import glob
import numpy as np
from PIL import Image
import pathlib
from keras.preprocessing.image import ImageDataGenerator

# Define the paths to the dataset folders
directory = os.getcwd()

train_folder = pathlib.Path(directory + '/cylinder/train')
val_folder = pathlib.Path(directory + '/cylinder/val')
test_folder = pathlib.Path(directory + '/cylinder/test')

# Define batch size
batch_size = 25

# Image preprocessing using ImageDataGenerator
train_image_generator = ImageDataGenerator(rescale=1./255)
val_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

# Helper function to load and preprocess an image
def preprocess_image(image_path, image_generator):
    image = Image.open(image_path)
    image = image.resize((128, 128))  # Adjust the desired width and height
    image = np.array(image)
    image = image_generator.random_transform(image)
    return image

# Helper function to read and process the text file
def preprocess_text_file(text_file_path):
    with open(text_file_path, 'r') as file:
        # Read and process the contents of the text file
        text_data = file.read()
        # Perform any required preprocessing steps on the text data
        # e.g., tokenization, cleaning, etc.
    return text_data

# Generator function to load and preprocess data in batches
def data_generator(file_paths, preprocess_func, *args):
    num_files = len(file_paths)
    for i in range(0, num_files, batch_size):
        batch_files = file_paths[i:i+batch_size]
        batch_data = []
        for file_path in batch_files:
            if preprocess_func.__name__ == 'preprocess_image':
                data = preprocess_func(file_path, *args)
            else:
                data = preprocess_func(file_path)
            batch_data.append(data)
        yield batch_data

""" # Function to prepare the dataset
def prepare_dataset(dataset_folder, image_generator):
    image_paths = glob.glob(os.path.join(dataset_folder, "**/*.png"), recursive=True)
    annotation_paths = glob.glob(os.path.join(dataset_folder, "**/*_crop_info.txt"), recursive=True)

    images = []
    annotations = []

    for image_path in image_paths:
        image_batch = preprocess_image(image_path, image_generator)
        images.append(image_batch)

    for annotation_path in annotation_paths:
        annotation_batch = preprocess_text_file(annotation_path)
        annotations.append(annotation_batch)

    # Convert the lists to NumPy arrays
    images = np.array(images)
    #rotations = np.array(rotations)
    #translations = np.array(translations)
    annotations = np.array(annotations)

    #print("Image paths:")
    #print(image_paths)
    #print("Annotation paths:")
    #print(annotation_paths)
    #print(images[-1])
    print(annotations[-1])
    print("Images shape:", images.shape)
    print("Annotations shape:", annotations.shape)
    return images, annotations """


def prepare_dataset(dataset_folder, image_generator):
    image_paths = glob.glob(os.path.join(dataset_folder, "**/*.png"), recursive=True)
    annotation_paths = glob.glob(os.path.join(dataset_folder, "**/*_crop_info.txt"), recursive=True)

    images = []
    annotations = []

    for image_path in image_paths:
        rotation_path = glob.glob(os.path.join(dataset_folder, "**/rotation.txt"), recursive=True)
        translation_path = glob.glob(os.path.join(dataset_folder, "**/translation.txt"), recursive=True)
        if  not image_path.endswith(".png"):
            print("Skipping file:", image_path, "- Unsupported format")
            continue
        print("x")
        
        if os.path.exists(rotation_path) or os.path.exists(translation_path):
            image_batch = preprocess_image(image_path, image_generator)
            images.append(image_batch)
        else:
            print("Skipping file:", image_path, "- Missing rotation or translation file")

    for annotation_path in annotation_paths:
        annotation_batch = preprocess_text_file(annotation_path)
        annotations.append(annotation_batch)

    images = np.array(images)
    annotations = np.array(annotations)

    print("Images shape:", images.shape)
    print("Annotations shape:", annotations.shape)

    return images, annotations


print("xx")
train_images, train_annotations = prepare_dataset(train_folder, train_image_generator)
print("xx")
val_images, val_annotations  = prepare_dataset(val_folder, val_image_generator)
print("xx")
test_images, test_annotations  = prepare_dataset(test_folder, test_image_generator)
print("xx")
# Save the training, validation, and testing datasets
np.save('train_images.npy', train_images)
np.save('train_annotations.npy', train_annotations)
np.save('val_images.npy', val_images)
np.save('val_annotations.npy', val_annotations)
np.save('test_images.npy', test_images)
np.save('test_annotations.npy', test_annotations)
