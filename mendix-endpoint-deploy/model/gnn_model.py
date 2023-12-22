import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

num_features = 10
num_classes = 40

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
            x = F.dropout(x, p=0.1, training=self.training)

        # Output layer
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
    

def save_model(model: torch.nn.Module, save_path):
    torch.save(model.state_dict(), save_path)

def load_model(model_path):
    model = CustomGraphSAGE(in_channels=num_features,
                            hidden_channels=16,
                            out_channels=num_classes,
                            num_layers=8)
    model.load_state_dict(torch.load(model_path))
    return model