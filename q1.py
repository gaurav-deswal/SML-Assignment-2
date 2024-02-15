import numpy as np
import matplotlib.pyplot as plt
import time

sample_no = 0

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
        
def visualize_training_samples(train_images, train_labels):
    try:
        if train_images.ndim != 3 or train_labels.ndim != 1:
            raise ValueError("ERROR: train_images must be a 3-dimensional array and train_labels must be a 1-dimensional array.")
        
        # Ensure the number of images and labels match
        if train_images.shape[0] != train_labels.shape[0]:
            raise ValueError("ERROR: The number of images and labels must match.")
        
        # Creates a figure and a grid of subplots with 10 rows and 5 columns, setting the figure size to 6x6 inches. This grid is for displaying 5 samples from each of the 10 classes (digits).
        fig, axes = plt.subplots(10, 5, figsize=(6, 6))
        
        # Flattens the 2D array of axes into a 1D array to make it easier to index into them in a linear fashion.
        axes = axes.flatten()
        
        for i in range(10):  # Loop over each class
            
            # Selects all images belonging to the current class i by indexing train_images with a boolean array where train_labels equals i.
            class_images = train_images[train_labels == i]
            
            if len(class_images) < 5:
                raise ValueError(f"Not enough samples for class {i} to display.")
            
            # Randomly select any 5 indexes of images
            indexes = np.random.choice(range(len(class_images)), 5, replace=False)
            
            for j, index in enumerate(indexes): # Loop over the selected sample indexes
                ax = axes[i*5 + j]  # Calculate the position of the subplot
                ax.imshow(class_images[index], interpolation='nearest') # Display the selected image at index 'index'
                ax.axis('off')
                # Set the title of the first subplot in each row to indicate the class (digit) being displayed. This helps identify which row corresponds to which digit.
                if j == 0:
                    ax.set_title(f'Class: {i}')
        
        plt.tight_layout() # Adjusts the layout of the subplots to ensure that they fit well within the figure area without overlapping.
        plt.show() # Displays the figure with the plotted images.
    except ValueError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Function computes the mean, covariance, and prior for each class.
def compute_class_statistics(images, labels):
     
    means = []
    covariances = []
    priors = []
    covariances_inv = []
    log_det_cov = []
    total_classes = 10
    regularization_value = 1e-6
    
    try:
        # Verify that the number of images and labels match
        if images.shape[0] != labels.shape[0]:
            raise ValueError("ERROR: The number of images does not match the number of labels.")
        
        # Calculate the mean, covariance and prior for each class
        for i in range(total_classes):
            
            class_samples = images[labels == i]
            # Ensure there are samples for the class
            if class_samples.size == 0:
                raise ValueError(f"ERROR: No samples found for class {i}.")
            
            mean = np.mean(class_samples, axis=0)
            
            if class_samples.shape[0] == 1:
                # If there is only one sample, we cannot compute a covariance matrix.
                raise ValueError(f"ERROR: Only one sample for class {i}. Hence, We can't compute covariance.")
            
            covariance = np.cov(class_samples, rowvar=False) + np.eye(class_samples.shape[1]) * regularization_value
            cov_inverse = np.linalg.inv(covariance)
            log_det = np.linalg.slogdet(covariance)[1]  # [1] is the log determinant
                
            
            means.append(mean)    
            covariances.append(covariance)
            covariances_inv.append(cov_inverse)
            log_det_cov.append(log_det)
            priors.append(class_samples.shape[0] / images.shape[0])
        
           
    except Exception as e:
        print(f"ERORR: An error occurred while computing class statistics: {e}")
        raise
            
    return means, covariances, covariances_inv, log_det_cov, priors

# Function calculates the QDA score for a given sample and class parameters.
# Formula- gi(x)= −(1/2)ln∣Σi∣ − (1/2)[(x - μi)T Σi-1(x - μi)] + lnP(ωi)
# where-
# • Σi is the covariance matrix of class i,
# •	μi is the mean vector of class i,
# •	x is the feature vector of the sample being classified,
# •	P(ωi) is the prior probability of class xi.
def qda_score(x, μ, Σ, Σ_inv, log_det_cov, prior):
    
    global sample_no
    sample_no += 1    
    print(f"QDA Score Computation for Sample #{sample_no}")
    
    try:
        # Compute the discriminant score       
        part1 = -0.5 * log_det_cov
        part2 = -0.5 * np.dot(np.dot((x - μ).T, Σ_inv), (x - μ))
        part3 = np.log(prior)
        
        return part1 + part2 + part3
    
    except np.linalg.LinAlgError as e:
        print(f"ERROR: Regularized Covariance Matrix is still non-invertible: {e}")
        raise
    except RuntimeWarning as e:
        print(f"WARNING: A runtime warning occurred in QDA discriminant function: {e}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in QDA discriminant function: {e}")
        raise

def classify_samples(samples, means, covariances, covariances_inv, log_det_cov, priors):
    try:
        predicted_classes = []
        start_time = time.time()  # Start timing
        for x in samples:
            scores = [qda_score(x, mean, cov, cov_inv, log_det, prior) for mean, cov, cov_inv, log_det, prior in zip(means, covariances, covariances_inv, log_det_cov, priors)]
            predicted_classes.append(np.argmax(scores)) # Store the highest qda values
    
    except Exception as e:
        print(f"ERROR: An error occurred during classification: {e}")
        raise
    elapsed_time = time.time() - start_time  # Stop timing
    print(f"\nClassification completed in {elapsed_time:.2f} seconds.")
    return np.array(predicted_classes)

def calculate_accuracy(predicted_labels, true_labels):
    try:
        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy
    except Exception as e:
        print(f"An error occurred while calculating accuracy: {e}")
        raise

def calculate_class_wise_accuracy(predicted_labels, true_labels, num_classes=10):
    try:
        class_wise_accuracy = []
        for i in range(num_classes):
            class_mask = true_labels == i
            correct_predictions = np.sum(predicted_labels[class_mask] == true_labels[class_mask])
            total_predictions = np.sum(class_mask)
            class_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            class_wise_accuracy.append(class_accuracy)
        return class_wise_accuracy
    except Exception as e:
        print(f"ERROR: An error occurred while calculating class-wise accuracy: {e}")
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

        # Assuming train_images and train_labels are loaded and valid
        visualize_training_samples(train_images, train_labels)   
        
        # Assuming train_images are loaded with shape (60000, 28, 28)

        # Vectorize the training images
        train_images_vectorized = train_images.reshape(train_images.shape[0], -1)

        # Checking the shape of the vectorized images
        print("\nShape of vectorized training images:", train_images_vectorized.shape)  # Expected: (60000, 784) 
        
        # Compute class statistics (means, covariances, and priors)
        means, covariances, covariances_inv, log_det_cov, priors = compute_class_statistics(train_images_vectorized, train_labels)
        
        # Vectorize the testing images
        test_images_vectorized = test_images.reshape(test_images.shape[0], -1)
        
        # Checking the shape of the vectorized images
        print("Shape of vectorized testing images:", test_images_vectorized.shape)  # Expected: (10000, 784) 

        # Classify the test samples
        predicted_test_classes = classify_samples(test_images_vectorized, means, covariances, covariances_inv, log_det_cov, priors)
        
        # Calculate overall accuracy
        overall_accuracy = calculate_accuracy(predicted_test_classes, test_labels)
        print(f"Overall test accuracy: {overall_accuracy * 100:.2f}%")
        
        # Calculate class-wise accuracy
        class_wise_accuracy = calculate_class_wise_accuracy(predicted_test_classes, test_labels)
        for i, accuracy in enumerate(class_wise_accuracy):
            print(f"Class {i} accuracy: {accuracy * 100:.2f}%")
        
        
    except Exception as e:
        print(f"ERROR: An error occurred in the main function: {e}")

main()