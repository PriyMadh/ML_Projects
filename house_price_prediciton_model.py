import numpy as np

x_train  = np.array([[4, 3, 2.53 ], [5, 6, 4.24 ],[5, 3, 2.55 ], [4, 3, 2.83 ], [3, 2, 2.01], [4, 2, 2.04]])
y_train = np.array([4.28, 7.00, 3.60, 5.29, 4.00, 2.80])

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost  = 0.0
    tot_cost = 0.0
    for i in range(0, m):
        f_wb_i = np.dot(x[i], w) + b
        cost = (f_wb_i - y[i])**2
        tot_cost = tot_cost+cost
    
    tot_cost = tot_cost / (2*m)
    return tot_cost

def compute_gradient_derivative(x, y, w, b ):
    m, n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0
    for i in range(m):
        tmp = (np.dot(x[i] , w)+ b) - y[i]
        for j in range (n):
            dj_dw[j] = dj_dw[j] + (tmp * x[i, j])
        dj_db = dj_db + tmp
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iterations, cost_function, gradient_function):
    J_history = []
    w = w_in.copy()
    b = b_in
    for i in range(num_iterations):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            if i%1000 == 0:
                print(f"cost at {i} interval is {J_history[i]}")
    
    return w, b, J_history

m, n  = x_train.shape
initial_w = np.zeros(n)
initial_b = 0.0
iterations = 100000
alpha = 1.0e-2
w_final, b_final, J_hist = gradient_descent(x_train, y_train, initial_w, initial_b, alpha, iterations, compute_cost, compute_gradient_derivative)

print(f"w_final : {w_final}, b_final: {b_final}")


