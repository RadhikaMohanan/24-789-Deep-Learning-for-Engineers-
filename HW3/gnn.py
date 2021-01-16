import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GCNConv, ChebConv,global_mean_pool,global_add_pool 
device = torch.device('cpu')

def get_dataset(save_path):
	'''
	read data from .npy file 
	no need to modify this function
	'''
	raw_data = np.load(save_path, allow_pickle=True)
	dataset = []
	for i, (node_f, edge_index, edge_attr, y)in enumerate(raw_data):
		sample = Data(
			x=torch.tensor(node_f, dtype=torch.float),
			y=torch.tensor([y], dtype=torch.float),
			edge_index=torch.tensor(edge_index, dtype=torch.long),
			edge_attr=torch.tensor(edge_attr, dtype=torch.float)
		)
		dataset.append(sample)
	return dataset


class GraphNet(nn.Module):

# '''
# Graph Neural Network class
# '''
    def __init__(self, n_features):
    # '''
		# n_features: number of features from dataset, should be 37
        # '''
        super(GraphNet, self).__init__()
		# define your GNN model here
        self.conv1 = GCNConv(n_features, 512, cached=False)
        self.bn1 = BatchNorm(512)
        self.conv2 = GCNConv(512,256, cached=False )
        self.bn2 = BatchNorm(256)
        self.conv3 = GCNConv(256,128, cached=False)
        self.bn3 = BatchNorm(128)
        self.conv4 = GCNConv(128,64, cached=False )
        self.bn4 = BatchNorm(64)
        self.conv5 = GCNConv(64,32, cached=False)
        self.bn5 = BatchNorm(32)
        self.conv6 = GCNConv(32,16, cached=False )
        # self.conv4 = GCNConv(8,4, cached=False )
        self.fc1 = Linear(16,8)
        self.fc2 = Linear(8,1)
		# raise NotImplementedError
		
    def forward(self, data):
		# define the forward pass here
		# raise NotImplementedError
        x, edge_index = data.x, data.edge_index
        x = F.leaky_relu(self.conv1(x, edge_index))
        x=self.bn1(x)
        x = F.leaky_relu(self.conv2(x, edge_index))
        x=self.bn2(x)
        x = F.leaky_relu(self.conv3(x, edge_index))
        x=self.bn3(x)
        x = F.leaky_relu(self.conv4(x, edge_index))
        x=self.bn4(x)
        x = F.leaky_relu(self.conv5(x, edge_index))
        x=self.bn5(x)
        x = F.leaky_relu(self.conv6(x, edge_index))
        
		# x = F.leaky_relu(self.conv4(x, edge_index))
        x = global_mean_pool(x,data.batch)
        x=F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


	

def main():
	# load data and build the data loader
    train_set = get_dataset('train_set.npy')
    test_set = get_dataset('test_set.npy')
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

	# number of features in the dataset
	# no need to change the value
    n_features = 37

	# build your GNN model
    model = GraphNet(n_features)

	# define your loss and optimizer
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(model)
    hist = {"train_loss":[], "test_loss":[]}
    num_epoch = 50
    for epoch in range(1, 1+num_epoch):
        model.train()
        loss_all = 0
        for data in train_loader:
			# your codes for training the model
			# ...
            optimizer.zero_grad()
            pred = model(data)
            y = data.y
            pred = pred.squeeze()
            loss = loss_func(pred,y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs * len(data)
            optimizer.step()
        train_loss = loss_all / len(train_set)
        
        with torch.no_grad():
            loss_all = 0
            for data in test_loader:
				# your codes for validation on test set
                pred2 = model(data)
                pred2 = pred2.squeeze()
                y = data.y
                loss = loss_func(pred2,y)
                loss_all += loss.item() * data.num_graphs * len(data)
            test_loss = loss_all / len(test_set)
            hist["train_loss"].append(train_loss)
            hist["test_loss"].append(test_loss)
            print(f'Epoch: {epoch}, Train loss: {train_loss:.3}, Test loss: {test_loss:.3}')

	# test on test set to get prediction 
    with torch.no_grad():
        prediction = np.zeros(len(test_set))
        label = np.zeros(len(test_set))
        idx = 0
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            prediction[idx:idx+len(output)] = output.squeeze().detach().numpy()
            label[idx:idx+len(output)] = data.y.detach().numpy()
            idx += len(output)
        prediction = np.array(prediction).squeeze()
        label = np.array(label).squeeze()
        tot = np.square(prediction-label).sum()
    print(tot)
    #save model
    torch.save(model.state_dict(), "p1_model.pkl")
    # visualization
    # plot loss function
    ax = plt.subplot(1,1,1)
    ax.plot([e for e in range(1,1+num_epoch)], hist["train_loss"], label="train loss")
    ax.plot([e for e in range(1,1+num_epoch)], hist["test_loss"], label="test loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    ax.legend()
    plt.show()

	# plot prediction vs. label
    x = np.linspace(np.min(label), np.max(label))
    y = np.linspace(np.min(label), np.max(label))
    ax = plt.subplot(1,1,1)
    ax.scatter(prediction, label, marker='+', c='red')
    ax.plot(x, y, '--')
    plt.xlabel("prediction")
    plt.ylabel("label")
    plt.show()


if __name__ == "__main__":
	main()
