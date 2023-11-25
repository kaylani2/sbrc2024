import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Get 9 random indices
random_indices = np.random.choice(len(x_test), 9, replace=False)

# Plot 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(6, 6))

for i, ax in enumerate(axes.flat):
  idx = random_indices[i]
  image = x_test[idx]
  label = y_test[idx]
  
  ax.imshow(image, cmap='gray')
  ax.set_title(f"Amostra: {label}")
  ax.axis('off')

plt.tight_layout()
#plt.show()
plt.savefig ('mnist_samples.pdf')


class_counts = Counter(y_train)
for label, count in class_counts.items():
    print(f"(Training set) Class {label}: {count} samples")

class_counts = Counter(y_test)
for label, count in class_counts.items():
    print(f"(Test set) Class {label}: {count} samples")
