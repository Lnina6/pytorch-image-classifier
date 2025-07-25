# pytorch-image-classifier
1. Data Loading & Preprocessing
  Uses the FashionMNIST dataset (a dataset of grayscale clothing images: shirts, shoes, bags, etc.).
  Automatically downloads the data and applies transforms.ToTensor() to convert images into normalized PyTorch tensors.
  The dataset is split into:
    Training set (75% of total training data)
    Validation set (25% of training data)
    Test set (held out for final evaluation)

2. DataLoader Setup
Wraps datasets using PyTorch DataLoader to:
  Shuffle the training data
  Provide mini-batches (e.g., 36 images per batch)
  Improve performance by avoiding manual loops

3. Model Architecture – SimpleCNN
A custom Convolutional Neural Network (CNN) with the following layers:
  Conv1: 1 input channel → 20 filters, kernel size 5
  Conv2: 20 → 46 filters, kernel size 4
  Conv3: 46 → 70 filters, kernel size 3
MaxPooling after each convolution
Fully Connected Layers:
  FC1: 70×3×3 → 512 neurons
  FC2: 512 → 10 (for the 10 clothing classes)

4. Loss Function & Optimizer
  CrossEntropyLoss is used for multi-class classification.
  SGD (Stochastic Gradient Descent) with momentum is used to update model weights.
  L2 regularization (weight decay) is added to reduce overfitting.

5. Training Loop
  Runs for a fixed number of epochs (e.g., 10).
  For each batch:
    Clears gradients
    Forward pass through the CNN
    Calculates the loss
    Backpropagates gradients
    Updates weights using the optimizer
    Tracks training loss and logs it with TensorBoard.

6. Validation Loop
  After each training epoch:
  Switches model to evaluation mode (model.eval())
  Disables gradient tracking to save memory
  Runs validation data through the model
Collects predictions and calculates:
  Validation los
  Accuracy
  Precision
  Recall
  F1 Score
  Stores metrics for later visualization

7. Visualization
  Plots the following after all epochs:
    Training and validation loss curves
    Accuracy over time
    Precision, Recall, and F1 score curves
    Creates a confusion matrix to show class-wise performance.
    Displays misclassified images with predicted vs true labels for interpretability.

8. Final Testing
  After training, the model is tested on the test set.
  Calculates final performance metrics.
  Shows another confusion matrix and some misclassified test examples.
