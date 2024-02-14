import numpy as np

def load_mnist_dataset(file_path):
    try:
        # Attempt to load the dataset
        with np.load(file_path) as data:
            train_images = data['x_train']
            train_labels = data['y_train']
            test_images = data['x_test']
            test_labels = data['y_test']
        # Successfully loaded the dataset
        print("MNIST dataset successfully loaded.")
        return train_images, train_labels, test_images, test_labels
    except FileNotFoundError:
        # The file was not found
        print(f"ERROR: The file '{file_path}' does not exist. Please check the file name and path.")
    except PermissionError:
        # Permission denied error
        print(f"ERROR: Permission denied when trying to read '{file_path}'. Please check the file permissions.")
    except KeyError as e:
        # Handling missing keys in the dataset file
        print(f"ERROR: The required data '{e}' is missing in the file. Please ensure the file has this key.")
    except Exception as e:
        # Generic catch-all for other exceptions
        print(f"An unexpected error occurred: {e}")

# Replace 'mnist.npz' with the path to MNIST dataset file
file_path = 'mnist.npz'
train_images, train_labels, test_images, test_labels = load_mnist_dataset(file_path)

# Check the shape of the data
print("Training images shape:", train_images.shape)  # Should be (60000, 28, 28)
print("Training labels shape:", train_labels.shape)  # Should be (60000,)
print("Test images shape:", test_images.shape)       # Should be (10000, 28, 28)
print("Test labels shape:", test_labels.shape)       # Should be (10000,)