import torch
import os
import json
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from gnn_model import load_model
import io
import pandas as pd
from torch_geometric.utils.convert import from_networkx, to_networkx

MODEL_FILE_NAME = 'GNN_ID_best_model.pth'


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model", MODEL_FILE_NAME)
    model = load_model(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        
        string_io = io.StringIO(request_body)
        df = pd.read_json(string_io)
        
        individual_graphs_data = []
        for k,v in df.groupby('graphId', sort=False):
            all_node_features = []
            all_node_labels = []
            
            node_id_type_map = {}

            for index, row in v.iterrows():
                node_id_type_map[row['originNodeId']] = row['originNodeType']
                node_id_type_map[row['destinationNodeId']] = row['destinationNodeType']
            
            index_of_UNK = list(node_id_type_map.values()).index('UNK')
            node_id_UNK = list(node_id_type_map.keys())[index_of_UNK]
            
            G = nx.DiGraph()  # Create a new graph for this 'graphId'
            edges = [(src, dst) 
                     for src, dst in zip(v["originNodeId"].tolist(), 
                                         v["destinationNodeId"].tolist())]
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
                all_node_labels.append(node_id_type_map.get(node_id))
            
            x = torch.tensor(np.stack(all_node_features), dtype=torch.float)
            edge_index = G.edge_index
            y = np.stack(all_node_labels)
            data = Data(x=x,
                           edge_index=edge_index,
                           y=y,
                           transform=NormalizeFeatures())
            
            data.unk_index = index_of_UNK
            data.unk_node_id = node_id_UNK
            data.unk_graph_id = k
            
            individual_graphs_data.append(data)

        return individual_graphs_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    pred = []
    for data in input_data:
        with torch.no_grad():
            data = data.to(device)
            out = model(data.x, data.edge_index)
            predictions = f"N{out.argmax(dim=1)[data.unk_index].item()}"
            pred.append({'graphId': data.unk_graph_id, 'nodeId': data.unk_node_id, 'prediction': predictions})
    return pred


def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        res = prediction
        return json.dumps(res)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")