import os
import kagglehub
import pandas as pd
import numpy as np

#  Download the dataset using kaggleub
path = kagglehub.dataset_download("noeyislearning/framingham-heart-study")

#  Look inside the folder to get CSV file
files = os.listdir(path)
csv_files = [f for f in files if f.endswith('.csv')]

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in the downloaded directory: {path}")

# Pick the first CSV file found
csv_filename = csv_files[0]
csv_path = os.path.join(path, csv_filename)
print(f"Success! Found and loading file: {csv_filename}")

# Load data into a Pandas DataFrame
df = pd.read_csv(csv_path)

df = df.dropna()
# THE NUMPY CONVERSION:

# Get the target column and make it a NumPy array
y = df["TenYearCHD"].to_numpy()

# Drop  target column so only have features left (X) then convert to a NumPy array
X = df.drop(columns=["TenYearCHD"]).to_numpy()

# Check shape and if they are numpy
print("\n--- Matrix Verification ---")
print("Type of X:", type(X))  
print("Type of y:", type(y))  
print("Feature matrix (X) shape:", X.shape)
print("Target array (y) shape:", y.shape)

X = X/100

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

# Compute cost J of sigmoid functioin w/ specific weights
def compute_cost_reg(X, y, w, b, lambda_ = 1):
    cost = 0.0
    m, n  = X.shape

    for i in range (m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        loss = -y[i] * np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
        cost += loss
    cost  = cost / m

# make sure model doesn't overfit
    reg_cost = 0.0
    for j in range (n):
        reg_cost = reg_cost +(w[j]**2)
    reg_cost = reg_cost * lambda_ / (2*m)

    total_cost = reg_cost + cost
    return total_cost

# find partial derivatives of cost function with respect to w and b (using vectorization to be more efficient)
def compute_gradient_reg (X, y, w,b, lambda_ = 0.5):
    m, n  = X.shape
    dj_dw = np.zeros((n, ))
    dj_db = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        tmp = f_wb_i - y[i]
        
        
        dj_dw += tmp * X[i] # b/c this is a scalar multiplied by vector so not dot prodeuct just multiplication
        dj_db += tmp
        
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    for j in range (n):
        dj_dw[j]  = dj_dw[j] + ((lambda_  / m) * w[j])
    return dj_dw, dj_db

# acc run gradient descent
def gradient_descent(X, y, w_in, b_in, num_iter, alpha, lambda_ = 1 ):
    w = w_in.copy()
    b = b_in
    J_history = []
    for i in range (num_iter):
        dj_dw, dj_db = compute_gradient_reg(X, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            J_history.append(compute_cost_reg(X, y, w ,b , lambda_))
            if i%1000 == 0:
                print(f"cost at iteration {i} is {J_history[i]}")
    return w, b, J_history

m, n  = X.shape
alpha = 1.0e-2
iterations = 10000
w_initial = np.zeros((n, ))
b_initial = 0.0
w_final, b_final, J_hist = gradient_descent(X, y, w_initial, b_initial, iterations, alpha, lambda_ = 1)

w_final = w_final * 100
print(f"w_final : {w_final}, b_final: {b_final}")








