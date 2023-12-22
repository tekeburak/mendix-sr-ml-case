# Mendix Assignment
# GraphSAGE-based Node Prediction

## Overview

This repository contains an implementation of a GraphSAGE (Graph Sample and Aggregation) model for predictive node classification in directed acyclic graphs (DAGs). The objective is to predict missing nodes within the graph structures leveraging machine learning techniques.

## GraphSAGE Model
The GraphSAGE model employed in this project utilizes a sampling and aggregation technique for generating node embeddings within graphs. GraphSAGE is a framework for inductive representation learning on large graphs. GraphSAGE is used to generate low-dimensional vector representations for nodes.
Key components include:
- **Sampling**: Nodes are sampled from the graph's local neighborhoods.
- **Aggregation**: Information from sampled nodes aggregates to generate node representations.
- **Training**: The model learns to predict missing or unlabeled nodes by optimizing an objective function. This implementation uses a custom GraphSAGE architecture for the task of predicting missing nodes in DAGs.

### Feature Extraction from Graph Data using NetworkX
Explaining how to perform feature extraction from graph data contained in a DataFrame (`df_train`) representing multiple graphs. The steps involved are as follows:
1.  **Grouping by Graph ID**
```python
for k, v in df_train.groupby('graphId', sort=False):
# ... (code within the loop)
```
2.  **Building Graph Representation**
- A directed graph (`nx.DiGraph()`) is created for each graph ID from the grouped data.
- Edges are added to the graph based on the 'originNodeId' and 'destinationNodeId' columns in the DataFrame.
```python
G = nx.DiGraph()
edges = [(src, dst) for src, dst in zip(v["originNodeId"].tolist(), v["destinationNodeId"].tolist())] G.add_edges_from(edges)
```
3.  **Calculating Node-Centric Graph Features**
- Node attributes and centrality measures are computed using NetworkX functions on the created graph.
- Centrality measures such as degree centrality, closeness centrality, betweenness centrality, PageRank, eigenvector centrality, Katz centrality, and local clustering coefficients are computed for each node.
```python
degree_centrality = nx.degree_centrality(graph)
closeness_centrality = nx.closeness_centrality(graph)
betweenness_centrality = nx.betweenness_centrality(graph)
pagerank_centrality = nx.pagerank(graph)
# ... (other centrality measures)
```
4.  **Node Feature Construction**
- Features for each node are constructed using the calculated centrality measures and other node-related information.
- These features include centrality measures, degrees (in, out, total), and clustering coefficients for each node in the graph.
```python
for node_index, node_id in enumerate(G.label):
	features = [
		# ... (centrality measures)
		degree_centrality.get(node_index),
		closeness_centrality.get(node_index),
		betweenness_centrality.get(node_index),
		pagerank_centrality.get(node_index),
		# ... (other centrality measures)
		node_degrees.get(node_index),
		in_degrees.get(node_index),
		out_degrees.get(node_index),
		# ... (other features) ]
	all_node_features.append(features)
	all_node_labels.append(int(node_id_type_map.get(node_id)[1:]))
```
5.  **Creating PyTorch Geometric Dataset**
- The extracted features and labels are converted into PyTorch tensors (`x` for features, `y` for labels) to create a PyTorch Geometric dataset. - Graph edges (`edge_index`) are extracted from the NetworkX graph representation.

```python
x = torch.tensor(np.stack(all_node_features), dtype=torch.float)
edge_index = G.edge_index
y = torch.tensor(np.stack(all_node_labels), dtype=torch.long)
dataset = Data(x=x, edge_index=edge_index, y=y, transform=NormalizeFeatures())
```

### Training the Graph Neural Network Model
#### Dataset Splitting and Data Loaders
- The provided code first splits the dataset into training and validation subsets using `train_test_split`. The split is set to have 90% for training and 10% for validation.
- PyTorch Geometric `DataLoader` objects are then created for both the training and validation datasets. These loaders enable batching and shuffling of the data during training.
```python
train_dataset, val_dataset = train_test_split(individual_graphs_data, train_size=0.9, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
```
#### Model Definition and Optimization
- A Graph Neural Network model (`CustomGraphSAGE`) is initialized with specified parameters such as input channels, hidden channels, output channels, and the number of layers.
- An optimizer (Adam) and a loss function (Negative Log-Likelihood Loss) are defined for training the model.
```python
model = CustomGraphSAGE(in_channels=num_features, hidden_channels=16, out_channels=num_classes, num_layers=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()
```
#### Training Loop
- The training loop runs for a set number of epochs (in this case, 200 epochs).
- During each epoch, the model is trained on the training dataset using the defined DataLoader.
- The model is set to train mode (`model.train()`) and iterates through the batches in the training DataLoader. It performs a forward pass, computes the loss, calculates gradients, updates model parameters, and clears gradients for the next iteration.
- Additionally, the code computes the training and validation accuracy after each epoch.
```python
for epoch in range(1, 201):
    loss = train()
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state_dict = model.state_dict()
    print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
```
#### Model Saving
- If the validation accuracy improves during training, the code saves the state of the best-performing model.
```python
if best_model_state_dict:
    torch.save(best_model_state_dict, 'model/GNN_ID_best_model.pth')
```

### AWS SageMaker Inference Structure
#### `inference.py`
#### Model Loading and Initialization
- The `model_fn` function is responsible for loading the model during inference. It loads the model from the specified path using the `load_model` function and sets it to evaluation mode.
- The `input_fn` function preprocesses the incoming request data. It reads the JSON input, extracts graph data, and processes it to create PyTorch Geometric `Data` objects for inference.
- The input is grouped by `graphId`, nodes and their attributes are extracted, and a PyTorch Geometric `Data` object is created for each graph.
#### Inference
- The `predict_fn` function performs the actual inference. It iterates through each prepared `Data` object, sends it to the model for prediction, and retrieves the predictions.
- These predictions are formatted as a list of dictionaries containing graph ID, node ID, and the predicted label.
#### Output Serialization
- The `output_fn` function serializes the predictions into a JSON format for the response.

### Deployment Configuration for SageMaker
#### `deploy.py`
#### Model Upload to S3
- The code begins by defining a function `upload_file` that uploads the model to an S3 bucket. It compresses the model folder using `compress_folder` from `compress.py` and uploads the compressed file to the specified S3 bucket.

#### SageMaker Model Deployment
- A PyTorchModel object is created using the `sagemaker.pytorch.PyTorchModel` class. This defines the configurations for deploying the model on SageMaker.
- The model configuration includes details such as the entry point for inference (`inference.py`), the source directory (`source_dir`), IAM role (`role`), model data location in S3 (`model_data`), framework version, Python version, and environment variables.
- The deployment mode (`local_mode`) is set to `True` for deploying the model locally on the current instance for testing purposes.
- `model.deploy()` method is used to initiate the deployment. It specifies the instance configuration, serializer, and deserializer for handling input and output data during inference.
```python
predictor = model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)
```
### Local Endpoint Testing via POSTMAN
#### Example JSON Data
- The example JSON file includes multiple graph entries, each with specified `graphId`, `originNodeId`, `originNodeType`, `destinationNodeId`, `destinationNodeType`, and `label`. This JSON data format simulates the input structure expected by the deployed endpoint.
- 
```json
[
    {
        "graphId":"d6fb5be2-4cb0-4418-be9e-9bd3b88bb855",
        "originNodeId":"89713516-1d89-4711-ac7a-f9e8426eb14d",
        "originNodeType":"N19",
        "destinationNodeId":"b898bfea-5196-4a36-af29-3610cabe7f67",
        "destinationNodeType":"N31",
        "label":null
    },
    // ... (other similar entries)
]
```

#### POSTMAN Screenshot
![](/mendix-solution/POSTMAN.png)

#### POSTMAN Return Data 
- The return data from the POSTMAN request: 
```json
[ { "graphId": "d6fb5be2-4cb0-4418-be9e-9bd3b88bb855", "nodeId": "3cd5edc4-55f1-40db-95ef-5ebae76838ce", "prediction": "N5" } ]
```

### Challenges Faced

During the implementation, I encountered challenges with node embeddings extraction using the [node2vec](https://karateclub.readthedocs.io/en/latest/_modules/karateclub/node_embedding/neighbourhood/node2vec.html) from the Karate Club framework. Despite employing this method, the achieved accuracy did not meet the desired expectations.

### GNN Model Exploration

In the pursuit of better performance, I experimented with various Graph Neural Network (GNN) models, including GCNConv, GAT, and GATConv etc. Despite these trials, the GraphSAGE model emerged as the most effective and performed better in terms of accuracy.

### Validation Accuracy Calculation and Test Data Submission
After training and evaluating the model on the `validation.parquet` dataset, the achieved validation accuracy stands at 54.17%.

The code responsible for calculating the validation accuracy and preparing the submission for the `test.parquet` files is located in the `utils` folder (`utils/create_test_data.py` and `utils/val_acc.py`).

#### Test Data Submission
The test submission file is located of root folder of `mendix-solution` folder as `submission_test.parquet`.