import numpy as np

houses = np.random.randint(0, 100, size = (20, 2)) # 20 houses with x, y coordinates

k = 3
random_indices = np.random.choice(houses.shape[0], size = k, replace=False) # randomly choose 3 warehouse positions from the 20 house coordinates
warehouses = houses[random_indices].astype(float)

#---------- find the closest centroid to the trianing exapmple---------

def find_closest_centroids(X, centroids):
    k = centroids.shape[0] # number of centroids = nubmer of warehouses
    idx = np.zeros(X.shape[0], dtype = int) # index for every x training vector
    J_i = np.zeros(X.shape[0]) #store the costs per each iteration of i
    for i in range (X.shape[0]):
        distances = [] #store the distances # recreate this list for every training example
        for j in range(centroids.shape[0]):
            distance_ij = np.linalg.norm(X[i] - centroids[j])**2 #the distacne between this training example i and this centroid j run for every centroid for every traing example
            distances.append(distance_ij)
        idx[i] = np.argmin(distances) # finds the index of the centroid witht the least distacne from X[i] traing vectotr
        J_i[i] = (np.sum(distances[idx[i]]))
    J_total = np.sum(J_i) / X.shape[0]
    return idx, J_total


def compute_centroid_position(X, idx, K):
    m, n  = X.shape
    centroids = np.zeros((K, n)) # array with K centroids and n elements per vectotr to match x-vector shapes
    for k in range (K):
        points_k = X[idx==k] # find the examples from X that are closest to the centroid k and create a new matrix of these points
        if points_k.shape[0] > 0:
            centroids[k] = np.sum(points_k, axis = 0) / points_k.shape[0] # sum down each feature from the X-array and the divide each of those sums by the amount of points that are assigned to centroid k --> porduces a vectro
    return centroids

def run_kmeans(X, initial_centroids, max_iters = 20, K = warehouses.shape[0]):
    m, n = X.shape
    k = initial_centroids.shape[0]
    centroids = initial_centroids # so you can run frist iteration
    idx = np.zeros(m) # index for every training examples

    for iter in range (max_iters):
        idx, J_total = find_closest_centroids(X, centroids)
        centroids = compute_centroid_position(X, idx, K)

        if iter / (max_iters - 1) == 1:
            J_final = J_total

    return idx, centroids, J_final

def kmeans_initializer(X, K):
    randidx = np.random.permutation(X.shape[0])
    initial_centroids = X[randidx[:K]] # takes first K (3 here) examples from X (houses here)

    return initial_centroids

best_J = 10000000 # start with infinite cost
best_centroids = None
best_idx = None

num_trials = 100

for trial in range (num_trials):
    initial_centroids = kmeans_initializer(houses, warehouses.shape[0])

    idx, centroids, J_final = run_kmeans(houses, initial_centroids, max_iters= 20, K = warehouses.shape[0])

    if J_final < best_J:
        best_J = J_final
        best_centroids = centroids
        best_idx = idx

    print(f"J_final: {J_final}")

print(f"Lowest Final Cost (J): {best_J:.4f}")
print("\nBest Warehouse Locations:\n", best_centroids)
print ("\nBest index:", best_idx)

