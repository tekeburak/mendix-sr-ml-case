import pickle
import os
import networkx as nx
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.nn import SAGEConv
from torch_geometric.utils.convert import from_networkx, to_networkx

num_classes = 40
num_features = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # Save DataFrame to a JSON file
# df_train.to_json('dataset/training.json', orient='records', indent=4)
# df_val.to_json('dataset/validation.json', orient='records', indent=4)
# df_test.to_json('dataset/test.json', orient='records', indent=4)
    
# Read training.parquet file
training_file = "dataset/training.parquet"
df_train = pd.read_parquet(training_file, engine='fastparquet')
# Read validation.parquet file
val_file = "dataset/validation.parquet"
df_val = pd.read_parquet(val_file, engine='fastparquet')
# Read test.parquet file
test_file = "dataset/test.parquet"
df_test = pd.read_parquet(test_file, engine='fastparquet')

do_calculation = False
individual_graphs_data = []

class CustomGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(CustomGraphSAGE, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.05, training=self.training)

        # Output layer
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


if do_calculation:
    for k,v in df_train.groupby('graphId', sort=False):
        all_node_features = []
        all_node_labels = []
        all_edges_index = []
        
        node_id_type_map = {}

        for index, row in v.iterrows():
            node_id_type_map[row['originNodeId']] = row['originNodeType']
            node_id_type_map[row['destinationNodeId']] = row['destinationNodeType']
        
        G = nx.DiGraph()  # Create a new graph for this 'graphId'
        edges = [(src,
                  dst)
                 for src, dst in zip(v["originNodeId"].tolist(), v["destinationNodeId"].tolist())]
        G.add_edges_from(edges)
        
        nx.set_node_attributes(G,{n:{'label':n} for n in G.nodes()})
        G = from_networkx(G)
        graph = to_networkx(G)
        
        degree_centrality = nx.degree_centrality(graph)
        closeness_centrality = nx.closeness_centrality(graph)
        betweenness_centrality = nx.betweenness_centrality(graph)
        pagerank_centrality = nx.pagerank(graph)
        
        try:
            eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=10000)
        except nx.PowerIterationFailedConvergence:
            print("Convergence failed. Adjust algorithm parameters or check graph structure.")
        
        katz_centrality = nx.katz_centrality(graph)
        local_clustering_coefficient = nx.clustering(graph)
        node_degrees = dict(graph.degree())
        in_degrees = dict(graph.in_degree())
        out_degrees = dict(graph.out_degree())
        
        for node_index, node_id in enumerate(G.label):
            features = [
                degree_centrality.get(node_index),
                closeness_centrality.get(node_index),
                betweenness_centrality.get(node_index),
                pagerank_centrality.get(node_index),
                eigenvector_centrality.get(node_index),
                katz_centrality.get(node_index),
                local_clustering_coefficient.get(node_index),
                node_degrees.get(node_index),
                in_degrees.get(node_index),
                out_degrees.get(node_index),
            ]
            all_node_features.append(features)
            all_node_labels.append(int(node_id_type_map.get(node_id)[1:]))
            
        
        x = torch.tensor(np.stack(all_node_features), dtype=torch.float)
        edge_index = G.edge_index
        y = torch.tensor(np.stack(all_node_labels), dtype=torch.long)
        dataset = Data(x=x, 
                       edge_index=edge_index, 
                       y=y,
                       transform=NormalizeFeatures())
        
        individual_graphs_data.append(dataset)
        
    with open(os.path.join('processed_data', 'individual_graphs_int_ID.pkl'), 'wb') as file:
        pickle.dump(individual_graphs_data, file)

with open(os.path.join('processed_data', 'individual_graphs_int_ID.pkl'), 'rb') as file:
    individual_graphs_data = pickle.load(file)

# Splitting data into train and validation data
train_dataset, val_dataset = train_test_split(individual_graphs_data, train_size=0.9, random_state=42)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = CustomGraphSAGE(in_channels=num_features,
                        hidden_channels=16,
                        out_channels=num_classes,
                        num_layers=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

def train():
    model.train()
    running_loss = 0.0
    
    for data in train_loader:  # Iterate in batches over the training dataset.
         data = data.to(device)
         out = model(data.x, data.edge_index)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.
         
         running_loss += loss.item()
    
    return running_loss / train_loader.__len__()

def test(loader):
     model.eval()

     correct = 0
     total = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         data = data.to(device)
         out = model(data.x, data.edge_index)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
         total += len(data.y)
     return correct / total  # Derive ratio of correct predictions.

best_val_acc = 0.0
best_model_state_dict = None
for epoch in range(1, 201):
    loss = train()
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state_dict = model.state_dict()
    print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
if best_model_state_dict:
    torch.save(best_model_state_dict, 'model/GNN_ID_best_model.pth')