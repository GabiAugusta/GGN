import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import argparse
import time
import copy
from utils.model import *
from tools import *
import pickle
import os
import datetime
torch.set_default_tensor_type('torch.DoubleTensor')

# Training settings
parser = argparse.ArgumentParser(description='VAR Model Training')
parser.add_argument('--epoch-num', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--dynamics-steps', type=int, default=20,
                    help='number of steps for dynamics learning (default: 20)')
parser.add_argument('--reconstruct-steps', type=int, default=10,
                    help='number of steps for reconstruction (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=2050,
                    help='random seed (default: 2050)')

args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)
use_cuda = not args.no_cuda and torch.cuda.is_available()

# Training parameters
Epoch_Num = args.epoch_num
Batch_Size = args.batch_size
Dyn_Steps = args.dynamics_steps
Net_Steps = args.reconstruct_steps

print("Data Loading...")
# Load VAR model data
train_loader, val_loader, test_loader, random_adj_matrix = load_var_data(batch_size=Batch_Size, data_path='./var_data/')
print('Train set batch num:', len(train_loader))
print('Validation set batch num:', len(val_loader))
print('Test set batch num:', len(test_loader))
print('Batch size:', Batch_Size)

# Initialize dynamics learner and optimizer
num_nodes = random_adj_matrix.shape[0]
dyn_learner = GumbelGraphNetworkClf(2)
dyn_learner = dyn_learner.double()
if use_cuda:
    dyn_learner = dyn_learner.cuda()
optimizer_dyn = optim.Adam(dyn_learner.parameters(), lr=0.001)

# Initialize Gumbel generator and optimizer
gumbel_generator = Gumbel_Generator(sz=num_nodes, temp=10, temp_drop_frac=0.9999)
gumbel_generator = gumbel_generator.double()
if use_cuda:
    gumbel_generator = gumbel_generator.cuda()
optimizer_network = optim.Adam(gumbel_generator.parameters(), lr=0.1)

# Loss function
loss_fn = torch.nn.MSELoss()

# Save folder for logs and models
now = datetime.datetime.now()
timestamp = now.isoformat()
save_folder = f'./logs/exp-{timestamp}/'
os.makedirs(save_folder, exist_ok=True)

# Save paths
generator_file = os.path.join(save_folder, 'generator.pt')
dyn_file = os.path.join(save_folder, 'dyn_learner.pt')

# Training loop
for epoch in range(Epoch_Num):
    print(f'Epoch {epoch + 1}/{Epoch_Num}')

    # Train dynamics learner
    print("\n*************** Dynamics Training ******************")
    dyn_learner.train()
    for step in range(Dyn_Steps):
        for batch_idx, (data_train, data_target) in enumerate(train_loader):
            if use_cuda:
                data_train = data_train.cuda()
                data_target = data_target.cuda()
            optimizer_dyn.zero_grad()
            adj = gumbel_generator.sample(hard=True).double()  # Ensure adj is Double
            adj = adj.unsqueeze(0).repeat(data_train.size(0), 1, 1)
            output = dyn_learner(data_train, adj)
            loss = loss_fn(output, data_target)
            loss.backward()
            optimizer_dyn.step()
        print(f'Dynamics Step {step + 1}/{Dyn_Steps}, Loss: {loss.item()}')

    # Train Gumbel generator
    print("\n*************** Gumbel Training ******************")
    gumbel_generator.train()
    for step in range(Net_Steps):
        for batch_idx, (data_train, data_target) in enumerate(train_loader):
            if use_cuda:
                data_train = data_train.cuda()
                data_target = data_target.cuda()
            optimizer_network.zero_grad()
            adj = gumbel_generator.sample(hard=True)
            adj = adj.unsqueeze(0).repeat(data_train.size(0), 1, 1)
            output = dyn_learner(data_train, adj)
            loss = loss_fn(output, data_target)
            loss.backward()
            optimizer_network.step()
        print(f'Gumbel Step {step + 1}/{Net_Steps}, Loss: {loss.item()}')

    # Validation
    print("\n*************** Validation ******************")
    dyn_learner.eval()
    val_losses = []
    with torch.no_grad():
        for batch_idx, (data_val, data_target) in enumerate(val_loader):
            if use_cuda:
                data_val = data_val.cuda()
                data_target = data_target.cuda()
            adj = gumbel_generator.sample(hard=True)
            adj = adj.unsqueeze(0).repeat(data_val.size(0), 1, 1)
            output = dyn_learner(data_val, adj)
            val_loss = loss_fn(output, data_target)
            val_losses.append(val_loss.item())
    avg_val_loss = np.mean(val_losses)
    print(f'Validation Loss: {avg_val_loss}')

    # Save the best model
    if epoch == 0 or avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print("Saving best model...")
        torch.save(gumbel_generator.state_dict(), generator_file)
        torch.save(dyn_learner.state_dict(), dyn_file)

# Testing
print("\n*************** Testing ******************")
gumbel_generator.load_state_dict(torch.load(generator_file))
dyn_learner.load_state_dict(torch.load(dyn_file))
dyn_learner.eval()
test_losses = []
with torch.no_grad():
    for batch_idx, (data_test, data_target) in enumerate(test_loader):
        if use_cuda:
            data_test = data_test.cuda()
            data_target = data_target.cuda()
        adj = gumbel_generator.sample(hard=True)
        adj = adj.unsqueeze(0).repeat(data_test.size(0), 1, 1)
        output = dyn_learner(data_test, adj)
        test_loss = loss_fn(output, data_target)
        test_losses.append(test_loss.item())
avg_test_loss = np.mean(test_losses)
print(f'Test Loss: {avg_test_loss}')