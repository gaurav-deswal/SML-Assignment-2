import numpy as np
from q1 import load_mnist_dataset

def create_data_matrix(train_images, train_labels):
    try:
        num_samples_per_class = 100
        num_classes = 10
        image_size = 28 * 28  # Vectorized image size
        
        # Check if train_images and train_labels have expected shapes
        if train_images.ndim != 2 or train_labels.ndim != 1:
            raise ValueError("train_images must be 2-dimensional and train_labels must be 1-dimensional.")
        if train_images.shape[0] != train_labels.size:
            raise ValueError("The number of rows in train_images must match the size of train_labels.")
        if train_images.shape[1] != image_size:
            raise ValueError("Each image in train_images must have a size of 784 (28x28 pixels).")
        
        # Initialize the data matrix X with zeros
        X = np.zeros((image_size, num_samples_per_class * num_classes))
        
        for class_id in range(num_classes):
            # Step 1: Find indices of all samples belonging to class i
            class_indices = np.where(train_labels == class_id)[0]
            if class_indices.size < num_samples_per_class:
                raise ValueError(f"Not enough samples for class {class_id}. Needed {num_samples_per_class}, got {class_indices.size}.")
            
            # Step 2: Randomly select 100 unique samples from the identified class samples.
            # This ensures that we don't pick the same sample more than once.
            selected_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)
            
            # Step 3: Retrieve the selected samples.
            # This involves fetching the images based on the indices we've chosen.
            selected_samples = train_images[selected_indices]
            
            # Step 4: Reshape each selected sample to a flat vector.
            # Since each image is 28x28 pixels, we flatten it to a 784-element vector.
            flattened_samples = selected_samples.reshape(num_samples_per_class, image_size)        
            
            # Step 5: Place the flattened samples into the data matrix X.
            # We calculate the start and end column indices for the current class in the data matrix.
            start_col = class_id * num_samples_per_class
            end_col = (class_id + 1) * num_samples_per_class
            X[:, start_col:end_col] = flattened_samples.T  # Transpose to match the dimensions of X.
        
        return X
    except ValueError as e:
        print(f"ValueError: {e}")
        raise
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        raise

def remove_mean(X):
    try:
        # Ensure X is a 2-dimensional numpy array
        if X.ndim != 2:
            raise ValueError("Input X must be a 2-dimensional numpy array.")
        
        # Calculate the mean of each row (feature)
        mean_vector = np.mean(X, axis=1)
        
        # Check if mean_vector contains NaN values which can occur if X has NaN values
        if np.isnan(mean_vector).any():
            raise ValueError("NaN values found in X. Cannot compute mean.")
        
        # Subtract the mean from each element in the row
        X_centered = X - mean_vector[:, np.newaxis]
        
        return X_centered, mean_vector
    except ValueError as e:
        print(f"ValueError: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def main():
    
    try:
        # Replace 'mnist.npz' with the path to MNIST dataset file
        file_path = 'mnist.npz'
        train_images, train_labels, test_images, test_labels = load_mnist_dataset(file_path)

        # Check the shape of the data
        print("Training images shape:", train_images.shape)  # Should be (60000, 28, 28) <- There are 60K training images of size 28 pixels X 28 pixels
        print("Training labels shape:", train_labels.shape)  # Should be (60000,)
        print("Test images shape:", test_images.shape)       # Should be (10000, 28, 28)
        print("Test labels shape:", test_labels.shape)       # Should be (10000,)

        # Vectorized train_images to create vectors of size 784 from the original 28x28 images.
        train_images_vectorized = train_images.reshape(train_images.shape[0], -1).T  # Reshape to 784 x 60000
        
        # Checking the shape of the vectorized images
        print("\nShape of vectorized training images:", train_images_vectorized.shape)  # Expected: (60000, 784)
        
        # Create X to represent our 784x1000 data matrix
        X = create_data_matrix(train_images_vectorized.T, train_labels)  # Transposed train_images_vectorized to get 60000x784 as input
        print("Shape of data matrix X:", X.shape)  # Should print (784, 1000)
        
        # X is our data matrix from Step 1
        X_centered, mean_vector = remove_mean(X)
        print("Shape of centered data matrix X:", X_centered.shape)  # Should still be (784, 1000)

    except Exception as e:
        print(f"ERROR: An error occurred in the main function: {e}")

if __name__ == "__main__":
    # Code here will only execute when the module is run directly, not when imported
    main()