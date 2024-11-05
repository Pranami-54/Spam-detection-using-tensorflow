# Spam Email Detection Using TensorFlow

This project implements a spam email detection system using a deep learning model built with **TensorFlow** and **Keras**. The model classifies SMS or email messages as either "ham" (non-spam) or "spam". The project demonstrates how to preprocess text data, train a neural network, and evaluate its performance for spam detection.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The spam email detection system is built using **TensorFlow**. It uses the SMS Spam Collection dataset to train a binary classification model that can predict whether a message is "spam" or "ham".

Key steps involved:
1. Data loading and preprocessing
2. Text vectorization using **TF-IDF**
3. Model building using a simple feedforward neural network
4. Model training, evaluation, and prediction

## Dataset

The model is trained on the **SMS Spam Collection Dataset**, which contains 5,572 text messages labeled as either spam or ham. The dataset includes two columns:
- **Category**: Label (`ham` or `spam`)
- **Message**: The text of the message

## Requirements

This project requires the following Python libraries:
- `tensorflow`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`

To install the required libraries, you can use the following `pip` command:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

## Usage

### 1. Prepare your data

Ensure that the **SMS Spam Collection** dataset (`spam.csv`) is in the project directory. The dataset must have the following columns:
- `Category`: Label indicating whether the message is spam or ham.
- `Message`: The text of the SMS/email message.

### 2. Run the Model

To run the spam detection model, execute the following Python script:

```bash
python spam_detection.py
```

This will:
- Preprocess the data (encode labels, vectorize text).
- Split the data into training and test sets.
- Train a deep learning model.
- Plot the training and validation accuracy.
- Evaluate the model on the test data.
- Output predictions for new messages.

### 3. Test with New Messages

You can use the `predict_spam` function to classify new messages as spam or ham. For example:

```python
new_message = "You have won a free vacation!"
prediction = predict_spam(new_message)
print(f"The message is: {prediction}")
```

## Model Architecture

The spam detection model is a simple feedforward neural network with the following architecture:
1. **Input Layer**: Matches the number of features in the TF-IDF representation of the text (5000 features).
2. **Hidden Layer 1**: 64 neurons with ReLU activation.
3. **Hidden Layer 2**: 32 neurons with ReLU activation.
4. **Output Layer**: 1 neuron with a sigmoid activation function for binary classification (spam or ham).

The model is compiled with:
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

## Results

After training the model on the dataset, the performance is evaluated on the test set. The model provides accuracy scores for both training and test data.

For example, after 5 epochs, the model may achieve an accuracy of approximately **98%** on the test set (depending on the random split and training process).
