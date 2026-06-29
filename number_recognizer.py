import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt

# 1. Find in Mac

#csv_path = "/Users/priyanmadhani/Downloads/digit-recognizer/train.csv"
csv_path = "train.csv"


if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"Could not find the file at {csv_path}. Double check your Mac username spelling!"
    )

print(f"Success! Found and loading file from Finder: {csv_path}")

# 2. Load data into a Pandas DataFrame
df = pd.read_csv(csv_path)


df = df.dropna()

# 3. THE NUMPY CONVERSION:

# Get the target column 
y_train = df["label"].to_numpy()

# Drop target column 
X_raw = df.drop(columns=["label"]).to_numpy()

# Normalize the values 
X_train = X_raw / 255.0

# 4. Check shape and types
print("\n--- Matrix Verification ---")
print("Type of X:", type(X_train))
print("Type of y:", type(y_train))
print("Feature matrix (X) shape:", X_train.shape)  # Expected: (42000, 784)
print("Target array (y) shape:", y_train.shape)  # Expected: (42000,)

test_csv_path = "/Users/priyanmadhani/Downloads/digit-recognizer/test.csv"
test_df = pd.read_csv(test_csv_path)
X_test_raw = test_df.to_numpy()
X_test = X_test_raw / 255.0
print(f"X_test shape: {X_test.shape}")
print(f"X_test first value {X_test}")


tf.random.set_seed(1234) #less randomness
model = Sequential(
    [
        
        Dense(units = 56, activation = 'relu'),
        Dense(units = 25, activation = 'relu'),
        Dense(units = 10, activation = 'linear'),
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #less rounding error by using linear activation in output layer then using this loss
    optimizer = tf.keras.optimizers.SGD(learning_rate=5e-2), #use gradient descent function (tried ADAM before but didn't work so well. kept overshooting)
)

model.fit(X_train, y_train, epochs = 30)

for i in range (0, X_train.shape[0]):
    if i%420 == 0:     #Don't wanna run this for 42000 different images
        single_image_pixels = X_test[i]
        input_data = single_image_pixels.reshape(1, 784) #flatten it out to make a vector
        probabilities = tf.nn.softmax(model.predict(input_data)) #softmax ((z1 / z1+...+zn)) #multiclass
        prediction = np.argmax(probabilities)
        image_grid = single_image_pixels.reshape(28, 28) #back to image array 3d of pixels
        plt.figure(figsize = (5,5))
        plt.imshow(image_grid, cmap = 'gray')
        plt.title(f"Model Prediction: {prediction}", fontsize=16, color="blue") #show prediction as label
        plt.axis("off")
        plt.show()
