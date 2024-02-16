import numpy as np
import matplotlib.pyplot as plt
from q1 import load_mnist_dataset, calculate_accuracy, calculate_class_wise_accuracy


def create_data_matrix_and_labels(train_images, train_labels):
    try:
        num_samples_per_class = 100
        num_classes = 10
        image_size = 28 * 28  # Vectorized image size
        selected_indices_list = []  # We will store the indices of selected samples
        
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
            selected_indices_list.extend(selected_indices) # We will return this list of 100 selected indices
            
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
        
        labels_subset = train_labels[selected_indices_list]
        
        return X, labels_subset
    
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

def apply_pca(X_centered):
    try:
        # Ensure X_centered is 2-dimensional
        if X_centered.ndim != 2:
            raise ValueError("X_centered must be a 2-dimensional numpy array")

        # Step 1: Compute the Covariance Matrix
        num_samples = X_centered.shape[1]
        S = np.dot(X_centered, X_centered.T) / (num_samples - 1) # Covariance S = XX^T/999
        
        # Step 2: Compute the eigenvectors and eigenvalues of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(S)  # eigh is used for symmetric matrices like S
        
        # Step 3: Sort the eigenvectors by decreasing eigenvalues
        index = np.argsort(eigenvalues)[::-1]  # Get the indices that would sort eigenvalues in descending order
        sorted_eigenvalues = eigenvalues[index]
        sorted_eigenvectors = eigenvectors[:, index]
        
        # Step 4: Create matrix U from sorted eigenvectors
        U = sorted_eigenvectors
        
        return U, sorted_eigenvalues        
    except ValueError as e:
        print(f"ValueError: {e}")
        raise
    except np.linalg.LinAlgError as e:
        print(f"ERROR: Linear Algebra error during PCA. Failed to compute Eigenvalues and Eigenvectors: {e}")
        raise
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in the apply_pca function: {e}")
        raise

def reconstruct_data(U, X_centered):
    try:
        # Step 1: Perform Y = U^T X
        Y = np.dot(U.T, X_centered)
        
        # Step 2: Reconstruct X_recon = UY
        X_recon = np.dot(U, Y)
        
        return X_recon
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in the reconstruct_data function during data reconstruction: {e}")
        raise

def calculate_mse(X, X_recon):
    try:
        mse = np.mean((X - X_recon) ** 2)
        return mse
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in the calculate_mse function while calculating MSE: {e}")
        raise

def reconstruct_and_plot_images(U, X_centered, mean_vector, train_labels, num_components = 5):
    try:
        # Step 1: Select the first p eigenvectors from U to form Up
        Up = U[:, :num_components]
        
        # Step 2: Compute Y = Up^T X_centered
        Y = np.dot(Up.T, X_centered)
        
        # Step 3: Reconstruct the data X_recon = UpY
        X_recon = np.dot(Up, Y)
        
        # Step 4: Add back the mean to each feature
        X_recon += mean_vector[:, np.newaxis]
        
        # Plotting 5 images from each class
        fig, axes = plt.subplots(10, 5, figsize=(12, 12))
        
        for class_id in range(10):
            # Find indices of all samples belonging to the current class
            class_indices = np.where(train_labels == class_id)[0]
            if class_indices.size < 5:
                raise ValueError(f"Not enough samples for class {class_id} to display. Needed 5, got {class_indices.size}.")
            
            # Select distinct indices for plotting
            selected_indices = class_indices[:5]  # Select first 5 samples per class
            
            for i, idx in enumerate(selected_indices):
                # Plotting the reconstructed images
                img = X_recon[:, idx].reshape(28, 28)
                ax = axes[class_id, i]
                ax.imshow(img, interpolation='nearest') # For grayscale images
                ax.axis('off')
                if i == 0:
                    ax.set_title(f'Class {class_id}, p={num_components}')
        plt.suptitle(f'Reconstructed Images with p={num_components}', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    except ValueError as e:
        print(f"ValueError: {e}")
        raise  
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in reconstruct_and_plot_images function during image reconstruction and plotting for p={num_components}: {e}")
        raise

def compute_class_statistics_pca(Y_train, train_labels):
    total_classes = 10
    means = []
    covariances = []
    priors = []
    log_dets = []  # Store log determinants for each class
    regularization_value = 0.01
    
    # print("\nCOMPUTING CLASS STATISTICS")
    
    for i in range(total_classes):
        try:
            
            class_indices = np.where(train_labels == i)[0]
            # Ensure there are enough samples for the class
            if len(class_indices) == 0:
                raise ValueError(f"No samples found for class {i}.")
            
            if np.any(class_indices >= Y_train.shape[1]):
                raise ValueError("Class indices out of bounds.")
            
            class_samples = Y_train[:, class_indices]
            
            if class_samples.size == 0:
                raise ValueError(f"ERROR: No samples found for class {i}.")
            
            # Ensure class_samples is 2D with samples as columns
            if class_samples.ndim == 1:
                class_samples = class_samples[:, np.newaxis]
                
            mean = np.mean(class_samples, axis=1)
            covariance = np.cov(class_samples, bias=True) + np.eye(class_samples.shape[0]) * regularization_value
            prior = class_samples.shape[1] / Y_train.shape[1]
            
            # Compute the signed log determinant of the covariance matrix
            sign, log_det = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise np.linalg.LinAlgError("Covariance matrix is not positive definite.")
            
            means.append(mean)
            covariances.append(covariance)
            priors.append(prior)
            log_dets.append(log_det)
        except ValueError as e:
            print(f"ValueError for class {i}: {e}")
        except np.linalg.LinAlgError as e:
            print(f"Error computing class {i} statistics: {e}")
            raise
        except Exception as e:
            print(f"ERROR: An unexpected error occurred in compute_class_statistics_pca function:{e}")
        
    # print("COMPUTING CLASS STATISTICS COMPLETED")
    return means, covariances, priors, log_dets

def qda_score_pca(x, mean, covariance, prior, log_det):
    try:
        # print("\n COMPUTING qda score")
        cov_inv = np.linalg.inv(covariance)
        part1 = -0.5 * log_det
        part2 = -0.5 * np.dot(np.dot((x - mean).T, cov_inv), (x - mean))
        part3 = np.log(prior)
        # print("COMPUTING qda score COMPLETED")
        return part1 + part2 + part3
    except np.linalg.LinAlgError:
        print("Error inverting covariance matrix.")
        raise
    except Exception as e:
        print(f"Unexpected error in qda_score_pca: {e}")
        raise

def classify_samples_pca(Y_test, means, covariances, priors, log_dets):
    num_samples = Y_test.shape[1]
    num_classes = len(means)
    predicted_classes = np.zeros(num_samples)
    
    try:
        # print("CLASSIFYING SAMPLES")
        for i in range(num_samples):
            scores = []
            
            for j in range(num_classes):
                score = qda_score_pca(Y_test[:, i], means[j], covariances[j], priors[j], log_dets[j])
                scores.append(score)
            predicted_classes[i] = np.argmax(scores)
    except Exception as e:
        print(f"Error during classification: {e}")
        raise
    # print("CLASSIFYING SAMPLES COMPLETED")
    return predicted_classes

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
        print("\nShape of vectorized training images:", train_images_vectorized.shape)  # Expected: (784, 60000)
        
        # Create X to represent our 784x1000 data matrix
        X, subset_train_labels = create_data_matrix_and_labels(train_images_vectorized.T, train_labels)  # Transposed train_images_vectorized to get 60000x784 as input
        print("Shape of data matrix X:", X.shape)  # Should print (784, 1000)
        
        # Compute X_centered by removing mean from X
        X_centered, mean_vector = remove_mean(X)
        print("Shape of centered data matrix X:", X_centered.shape)  # Should still be (784, 1000)

        # Compute Principal Component matrix U (eigenvectors sorted by eigenvalues)
        U, sorted_eigenvalues = apply_pca(X_centered)
        
        print("\nPCA applied successfully.")
        # print(f"\nPrincipal Component Matrix U (Eigenvectors sorted by Eigenvalues): {U}")
        
        # X_recon = UY where Y = U^TX
        X_recon = reconstruct_data(U, X_centered)
        # Compute mse between X and X_recon and print the same. It should be close to 0.
        mse = calculate_mse(X_centered, X_recon)
        print(f"\nMSE between X and X_recon = {mse}\n")
        
        # Reconstruct and Plot Images
        for p in [5, 10, 20]: # Note as p value increases, reconstructed images should look more like their original counterparts
            reconstruct_and_plot_images(U, X_centered, mean_vector, train_labels, p)
        
        # Q2 Last bullet point impl below
        
        # Vectorize and center the test images
        test_images_vectorized = test_images.reshape(test_images.shape[0], -1).T
        # Create X to represent our 784x1000 data matrix
        X_test, subset_test_labels = create_data_matrix_and_labels(test_images_vectorized.T, test_labels)  # Transposed
        
        X_test_centered, mean_test_vector = remove_mean(X_test)
        
        # Perform dimensionality reduction on the test set and classify with QDA for various p values
        for p in [5, 10, 20]:
            print(f"\nUsing {p} principal components:")
            
            # Select the top-p components
            U_p = U[:, :p]     
            
            # Compute class statistics on the PCA-transformed training data
            Y_train = np.dot(U_p.T, X_test_centered)
            Y_train_labels = subset_train_labels
            
            # print(f"Shape of PCA-transformed training data Y_train: {Y_train.shape}")  # Should be (# components, 1000) or similar
            # # Verify that train_labels correspond to columns in Y_train
            # print(f"Subset train labels length: {len(Y_train_labels)}")
            
            means, covariances, priors, log_dets = compute_class_statistics_pca(Y_train, Y_train_labels)
            
            # Project the centered test set onto these components
            Y_test = np.dot(U_p.T, X_test_centered)
            
            # Verify the dimensions
            # print(f"Shape of projected test data Y_test: {Y_test.shape}") # (5,1000) is correct
            
        
            # print(f"Mean vector shape for class 0: {means[0].shape}")
            # print(f"Covariance matrix shape for class 0: {covariances[0].shape}")
            
            # Classify the projected test set
            predicted_classes = classify_samples_pca(Y_test, means, covariances, priors, log_dets)
            
            # Calculate and print the accuracy
            accuracy = calculate_accuracy(predicted_classes, subset_test_labels)
            print(f"Accuracy: {accuracy * 100:.2f}%")
            
            # Calculate and print class-wise accuracy
            class_wise_accuracy = calculate_class_wise_accuracy(predicted_classes, subset_test_labels)
            for i, acc in enumerate(class_wise_accuracy):
                print(f"Class {i} accuracy: {acc * 100:.2f}%")
        
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in the main function: {e}")

if __name__ == "__main__":
    # Code here will only execute when the module is run directly, not when imported
    main()