from pymilvus import (
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, connections,
    AnnSearchRequest, connections, WeightedRanker, RRFRanker
)

from gen_features import ResNetFeatureExtractor, CLIPFeatureExtractor

import os
import numpy as np
from PIL import Image
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread

def show_images_horizontally(folder, width=15, height=5):
    files = os.listdir(folder)
    list_of_files = [f'{folder}/res{i}.jpg' for i in range(len(files))]
    fig = figure(figsize=(width, height))  # Set the figure size here (width, height) in inches
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a = fig.add_subplot(1, number_of_files, i + 1)
        image = imread(list_of_files[i])
        imshow(image, cmap='Greys_r')
        axis('off')
        a.set_aspect('auto')

def search_image(imgname, text, outdir, k):
    os.makedirs(outdir, exist_ok=True) 
    connections.connect("default", host="10.102.6.136", port="19530")
    col_name = f'fashioniq'
    col = Collection(col_name)
    col.load()
    resnet_fe = ResNetFeatureExtractor()
    clip_fe = CLIPFeatureExtractor()


    featc = None
    featr = None

    resnet_search_params = {"metric_type": "IP", "params": {}}
    clip_search_params = {"metric_type": "IP", "params": {}}

    if imgname != None: 
        image = Image.open(imgname).convert('RGB')
        featr = resnet_fe(image).detach().cpu().numpy()
        featr = featr / np.linalg.norm(featr)
        resnet_req = AnnSearchRequest(featr, "resnet_vector", resnet_search_params, limit=k)

    if text != None:
        featc = clip_fe(text).detach().cpu().numpy()
        featc = featc / np.linalg.norm(featc)
        clip_req = AnnSearchRequest(featc, "clip_vector", clip_search_params, limit=k)

    if text == None:
        results = col.search(featr, anns_field="resnet_vector", param=resnet_search_params, limit=k, output_fields=['filename'])
    elif imgname == None:
        results = col.search(featc, anns_field="clip_vector", param=clip_search_params, limit=k, output_fields=['filename'])
    else:
        results = col.hybrid_search([resnet_req, clip_req], rerank=RRFRanker(), limit=k, output_fields=['filename'])

    for i, result in enumerate(results[0]):
        os.system(f'cp ./pics/{result.filename}  {outdir}/res{i}.jpg ')
    

if __name__ == '__main__':
    search_image("mysearch.png", None, "resnet", 10)
    search_image(None, "a black dress with white buttons.", "clip", 10)
    search_image("mysearch.png", "a black dress with white buttons.", "hybrid", 10)
