# Convolutional Neural Network (CNN)

## What is a Convolutional Neural Network?

A Convolutional Neural Network (CNN) is a type of deep learning model that is mainly used for working with images. It is designed to automatically learn patterns such as edges, shapes, textures, and objects directly from raw data. Instead of manually defining rules, the network learns useful features by itself during training.

CNNs are widely used in image recognition, medical imaging, self-driving cars, and cybersecurity. They are especially powerful when the data can be represented as a grid, such as an image or a matrix of numbers.

---

## How a CNN is Structured

A CNN is built from a few important types of layers that work together.

### Convolution Layer

The convolution layer applies small filters (also called kernels) to the input image. Each filter slides across the image and extracts specific features.

Simple visualization:

```
Input Image (5x5)

1 1 1 0 0
0 1 1 1 0
0 0 1 1 1
0 0 1 1 0
0 1 1 0 0

Filter (3x3)

1 0 1
0 1 0
1 0 1
```

As the filter moves across the image, it creates a new matrix called a feature map. Different filters detect different patterns.

### Pooling Layer

The pooling layer reduces the size of the feature map. This makes the model faster and helps reduce overfitting. The most common type is Max Pooling.

Example (2x2 Max Pooling):

```
Before:
2 1 3 0
1 5 2 1
0 1 3 2
2 2 1 0

After:
5 3
2 3
```

The highest value in each small region is selected.

### Fully Connected Layer

After several convolution and pooling steps, the data is flattened into a single vector. The fully connected layer then uses these learned features to make the final prediction.

---

## How Everything Flows Together

```
Input Image
     ↓
Convolution
     ↓
ReLU Activation
     ↓
Pooling
     ↓
Fully Connected Layer
     ↓
Output (Prediction)
```

The model learns by adjusting its internal weights during training to reduce prediction errors.

---

## A Practical Cybersecurity Example

CNNs are not only for normal images. In cybersecurity, network traffic or malware files can be converted into image-like data. For example, packet byte values can be reshaped into a 2D matrix. The CNN can then classify traffic as normal or malicious.

In this simple example, we simulate network traffic as 16x16 grayscale images.

---

## Generating the Dataset

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)

normal = np.random.normal(loc=0.3, scale=0.1, size=(500,16,16))
malicious = np.random.normal(loc=0.7, scale=0.1, size=(500,16,16))

X = np.concatenate([normal, malicious])
y = np.array([0]*500 + [1]*500)

X = X.reshape(-1,16,16,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

plt.imshow(X[0].reshape(16,16), cmap='gray')
plt.title('Sample Network Traffic Image')
plt.colorbar()
plt.show()
```

Here, normal traffic has lower average values and malicious traffic has higher average values. This simulates different behavior patterns.

---

## Building and Training the CNN

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(16,16,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
```

This model learns to separate normal and malicious traffic patterns.

---

## Visualizing Training Performance

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Validation'])
plt.show()
```

This graph shows how the model improves over time.

---

## Why CNNs Work Well in Cybersecurity

CNNs are powerful because they automatically learn patterns from raw data. They can detect hidden structures that traditional rule-based systems may miss. They also reduce the need for manual feature engineering.

In real-world cybersecurity systems, CNNs are used for malware image classification, intrusion detection systems, botnet detection, and spam filtering when text is transformed into matrix representations.

---

## Final Thoughts

A Convolutional Neural Network is a deep learning model designed to process grid-like data such as images. It uses convolution layers to extract features, pooling layers to reduce size, and fully connected layers to make predictions.

When applied to cybersecurity, CNNs can transform raw traffic or malware data into image-like representations and automatically detect malicious behavior. The example provided here is simple, easy to reproduce, and requir