import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report
from torch_geometric.data import Data
from models.Encoder import VIT_GIN_Parallel
from dataset import GraphDataset
from fieldroaddatapipeline.dataloader import FieldRoadDataLoader
import time
from collections import defaultdict
from utils.logger import Logger
logger = Logger(model_name="VIT_GIN_Parallel",dataset_kind="paddy_small")
logger.log_environment_info()
# Define paths for the test dataset
test_path=dict(
    gnss="../../../autodl-tmp/dataset/dataset_high/paddy_small/sampled_paddy_43",
    adj="../../../autodl-tmp/dataset/dataset_high/paddy_small/sampled_paddy_adj",
    json="../../../autodl-tmp/dataset/dataset_high/paddy_small/Non-Identically_Distributed_Coco/sampled_paddy_43_test.json"
)

# Create the test dataset and dataloader
test_dataset = GraphDataset(test_path, mode='predict', num_workers=12, max_len=1000, drop_rate=0)
test_loader = FieldRoadDataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
logger.log_dataset_info(None,None,test_dataset)
# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VIT_GIN_Parallel(
    img_size=43, 
    patch_size=1, 
    in_chans=1, 
    num_classes=2, 
    embed_dim=108, 
    depth=1,               
    num_heads=6, 
    mlp_ratio=4., 
    qkv_bias=True,              
    drop_rate=0., 
    attn_drop_rate=0., 
    drop_path_rate=0.1
).to(device)
logger.log_model_info(model)
# Load the trained model weights
model.load_state_dict(torch.load('./weights/model.pt'), strict=False)
# Start testing
model.eval()
all_predictions = []
all_labels = []
total_test_time = 0
trace_results = defaultdict(list)
num_samples = 0
with torch.no_grad():
    for batch_id, (points, labels, adjs, trace_id, coordinates) in enumerate(test_loader()):
        points = points.clone().detach().to(torch.float32).squeeze(0).to(device)
        labels = labels.clone().detach().to(torch.int64).squeeze().to(device)
        adjs = adjs.clone().detach().to(torch.float32).squeeze(0).to(device)
        coordinates = coordinates.clone().detach().to(torch.float64).squeeze().to(device)
        trace_id = trace_id[0]
        # 找到邻接矩阵中非零元素的索引
        rows, cols = torch.nonzero(adjs, as_tuple=True)
        # 按照源节点和目标节点的顺序构建新的张量
        edge_index = torch.stack([rows, cols]).to(device)
        data = Data(x=points, edge_index=edge_index, y=labels)
        predicts = model.test_step(data)
        predicts = torch.softmax(predicts, dim=1)
        trace_results[trace_id].append((points.cpu().numpy(), predicts.cpu().numpy(), labels.cpu().numpy(), coordinates.cpu().numpy()))
logger.log_start_of_outputs()
# Summarize results per trace_id
for trace_id, results in trace_results.items():
    points, predicts, labels, coordinates = zip(*results)
    
    points = np.concatenate(points, axis=0)
    predicts = np.concatenate(predicts, axis=0)
    labels = np.concatenate(labels, axis=0)
    coordinates = np.concatenate(coordinates, axis=0)
    prelabels = np.argmax(predicts, axis=1)
    class_result = classification_report(labels, prelabels, digits=4)
    logger.log_predict_info(coordinates, predicts[:,1],prelabels,labels, trace_id,class_result)
logger.clean_up_logger()