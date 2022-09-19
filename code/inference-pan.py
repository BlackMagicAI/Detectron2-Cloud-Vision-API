## https://github.com/aws/amazon-sagemaker-examples/blob/master/advanced_functionality/multi_model_pytorch/code/inference.py

import torch, torchvision
import sys
# from PIL import Image
from torchvision import transforms
import urllib
from urllib.request import urlopen
from io import BytesIO
# Some basic setup:
# Setup detectron2 logger

import detectron2
import detectron2.data.transforms as T
# # # import some common libraries
import numpy as np
import os, json, cv2, random

def model_fn(model_dir):
    model = torch.load(model_dir + "/model.pth")                  
    model.eval()
    return model

def input_fn(request_body, request_content_type):
   arr = np.asarray(bytearray(request_body), dtype=np.uint8) # convert to numpy.ndarray   
   data = cv2.imdecode(arr, -1)
   return data

def predict_fn(input_object, model):
    original_image = input_object
    aug = T.ResizeShortestEdge(
            [800, 800], 1333
        )
    predictions = None
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
#             if cfg.INPUT.FORMAT == "RGB":
#                 # whether the model expects BGR inputs or RGB
#                 original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = model([inputs])
    return predictions

def output_fn(predictions, content_type):
   instances = []
   for instance in predictions:
    # a Python object (dict):
    x = {"pred_boxes":instance['instances'].pred_boxes.tensor.cpu().numpy().tolist(),
        "pred_classes":instance['instances'].pred_classes.cpu().numpy().tolist(),
         "scores":instance['instances'].scores.cpu().numpy().tolist(),
         "panoptic_seg":instance['panoptic_seg'][0].cpu().numpy().tolist(),
         "instance_list":instance['panoptic_seg'][1]
        }
    # add to python list/array
    instances.append(x)
        # convert into JSON:
   return json.dumps(instances)
