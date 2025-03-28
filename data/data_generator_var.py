import numpy as np
import pickle
import os
import random

# Global configurations
NUM_VARIABLES = 5
TIME_STEPS = 100
BATCH_SIZE = 1024
SIGMA = 1.0  # Standard deviation for Gaussian noise
STORE_FOLDER = './var_data/'
AVERAGE_DEGREE = 2  # Average number of connections per node

# Ensure the store folder exists
os.makedirs(STORE_FOLDER, exist_ok=True)

# Generate the real adjacency matrix based on the VAR model
def generate_real_adj_matrix():
    adj_matrix = np.zeros((NUM_VARIABLES, NUM_VARIABLES))
    # Define the dependencies based on the VAR equations
    adj_matrix[0, [0, 4]] = 1  # x[0] depends on x[0] and x[4]
    adj_matrix[1, [1, 4]] = 1  # x[1] depends on x[1] and x[4]
    adj_matrix[2, [2, 3]] = 1  # x[2] depends on x[2] and x[3]
    adj_matrix[3, [0, 2, 3]] = 1  # x[3] depends on x[0], x[2], and x[3]
    adj_matrix[4, [3, 4]] = 1  # x[4] depends on x[3] and x[4]
    return adj_matrix

# Generate a random adjacency matrix
def generate_random_adj_matrix(num_variables, average_degree):
    adj_matrix = np.zeros((num_variables, num_variables))
    edges = []
    while len(edges) < num_variables * average_degree:
        start_node = random.randint(0, num_variables - 1)
        end_node = random.randint(0, num_variables - 1)
        if start_node != end_node and [start_node, end_node] not in edges:
            edges.append([start_node, end_node])
            adj_matrix[start_node, end_node] = 1
    return adj_matrix

# Generate synthetic data for the specified VAR model
def generate_var_data(num_variables, time_steps, batch_size, sigma):
    data = np.zeros((batch_size, num_variables, time_steps))
    for b in range(batch_size):
        x = np.zeros((num_variables, time_steps))
        eta = np.random.normal(0, sigma, (num_variables, time_steps))
        for t in range(4, time_steps):
            x[0, t] = 0.4 * x[0, t - 1] - 0.5 * x[0, t - 2] + 0.4 * x[4, t - 1] + eta[0, t]
            x[1, t] = 0.4 * x[1, t - 1] - 0.3 * x[1, t - 4] + 0.4 * x[4, t - 2] + eta[1, t]
            x[2, t] = 0.5 * x[2, t - 1] - 0.7 * x[3, t - 2] - 0.3 * x[2, t - 3] + eta[2, t]
            x[3, t] = 0.8 * x[3, t - 3] + 0.4 * x[0, t - 2] + 0.3 * x[2, t - 2] + eta[3, t]
            x[4, t] = 0.7 * x[4, t - 1] - 0.5 * x[4, t - 2] - 0.4 * x[3, t - 1] + eta[4, t]
        data[b] = x
    return data

# Save data and adjacency matrices
def save_data(data, real_adj_matrix, random_adj_matrix):
    mark = np.random.randint(0, 100000)
    data_address = STORE_FOLDER + f'mark-{mark}-var-data.pickle'
    real_adj_address = STORE_FOLDER + f'mark-{mark}-real-adjmat.pickle'
    random_adj_address = STORE_FOLDER + f'mark-{mark}-random-adjmat.pickle'

    with open(data_address, 'wb') as f:
        pickle.dump(data, f)

    with open(real_adj_address, 'wb') as f:
        pickle.dump(real_adj_matrix, f)

    with open(random_adj_address, 'wb') as f:
        pickle.dump(random_adj_matrix, f)

# Main function
if __name__ == "__main__":
    # Generate the real adjacency matrix
    real_adj_matrix = generate_real_adj_matrix()
    
    # Generate a random adjacency matrix
    random_adj_matrix = generate_random_adj_matrix(NUM_VARIABLES, AVERAGE_DEGREE)
    
    # Generate VAR data using the real adjacency matrix
    data = generate_var_data(NUM_VARIABLES, TIME_STEPS, BATCH_SIZE, SIGMA)
    
    # Save the data and adjacency matrices
    save_data(data, real_adj_matrix, random_adj_matrix)
    print("Data generation complete.")