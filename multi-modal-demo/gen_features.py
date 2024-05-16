import os
import torch

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from PIL import Image
import clip
import numpy as np

from pymilvus import (
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, connections,
)

class ResNetFeatureExtractor():
    def __init__(self):
        self.model = timm.create_model('resnet34', pretrained=True, num_classes=0, global_pool='avg')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))

    def __call__(self, image):
        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(image_tensor.to(self.device))
        return features

class CLIPFeatureExtractor():
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        self.model = model
        self.preprocess = preprocess

    def __call__(self, data):
        if type(data) is str:
            text = clip.tokenize([data]).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(text)
        else:
            img = self.preprocess(data).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(img)
        return features

def prepare_milvus(overwrite=True):         
    connections.connect("default", host="10.102.9.100", port="19530")
    col_name = f'fashioniq'

    if overwrite == True:
        utility.drop_collection(col_name)
    dim_resnet = 512
    dim_clip = 512
    fields = [
            # Use auto generated id as primary key
            FieldSchema(name="pk", dtype=DataType.VARCHAR,
                        is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="resnet_vector", dtype=DataType.FLOAT_VECTOR, dim=dim_resnet),
            FieldSchema(name="clip_vector", dtype=DataType.FLOAT_VECTOR, dim=dim_clip),
        ]

    schema = CollectionSchema(fields, "")
    col = Collection(col_name, schema, consistency_level="Strong")

    resnet_index = {"index_type": "FLAT", "metric_type": "IP"}
    col.create_index("resnet_vector", resnet_index)
    clip_index = {"index_type": "FLAT", "metric_type": "IP"}
    col.create_index("clip_vector", clip_index)
    return col

import time
def extract_features():
    col = prepare_milvus()
    filenames = os.listdir('./pics/')
    count = 0 
    resnet_fe = ResNetFeatureExtractor()
    clip_fe = CLIPFeatureExtractor()
    for filename in filenames: 
        if filename.endswith('.jpg') is False:
            continue
        count = count + 1 
        image_path = f'./pics/{filename}'
        image = Image.open(image_path).convert('RGB')
        
        
        feat1 = resnet_fe(image).detach().cpu().numpy()
        feat2 = clip_fe(image).detach().cpu().numpy()
       
        feat1 = feat1 / np.linalg.norm(feat1)
        feat2 = feat2 / np.linalg.norm(feat2)
    
        col.insert([[filename], [feat1.flatten()], [feat2.flatten()]])
    

if __name__ == '__main__':
    extract_features()
