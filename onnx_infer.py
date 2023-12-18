# Code credit: [FastSAM Demo](https://huggingface.co/spaces/An-619/FastSAM).

import torch
import numpy as np
from segment_anything.onnx import SamPredictorONNX
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

predictor = SamPredictorONNX('/edge_sam_3x_encoder.onnx', 'edge_sam_3x_decoder.onnx')

img='/home/local/sota/EdgeSAM/notebooks/images/truck.jpg'
image = cv2.imread(img)
predictor.set_image(image)


global_points=[[500, 375]]
global_point_label= [1]
global_points_np = np.array(global_points)[None]
global_point_label_np = np.array(global_point_label)[None]
masks, scores, _ = predictor.predict(
    point_coords=global_points_np,
    point_labels=global_point_label_np,
)
masks = masks.squeeze(0)
scores = scores.squeeze(0)

print(f'scores: {scores}')
area = masks.sum(axis=(1, 2))
print(f'area: {area}')

