import numpy as np
import pickle
import os

# Global configurations
NUM_VARIABLES = 5
TIME_STEPS = 100
BATCH_SIZE = 1024
SIGMA = 1.0  # Standard deviation for Gaussian noise
STORE_FOLDER = './var_data/'

# Ensure the store folder exists
os.makedirs(STORE_FOLDER, exist_ok=True)

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

# Save data
def save_data(data):
    mark = np.random.randint(0, 100000)
    data_address = STORE_FOLDER + f'mark-{mark}-var-data.pickle'

    with open(data_address, 'wb') as f:
        pickle.dump(data, f)

# Main function
if __name__ == "__main__":
    data = generate_var_data(NUM_VARIABLES, TIME_STEPS, BATCH_SIZE, SIGMA)
    save_data(data)
    print("Data generation complete.")