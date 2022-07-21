# import some common libraries
# Using Python Request library
import requests
import json
import numpy as np
import time
import io
from PIL import Image, ImageDraw, ImageFont

# Define Constants
IMAGE_PATH="<INSERT_IMAGE_FILE_PATH_HERE>"
API_INVOKE_URL="<INSERT_API_INVOKE_URL>"

# define variables
url=API_INVOKE_URL

#Used to print time lapsed during different parts of code execution.
def printTimeStamp(start, log):
  end = time.time()
  print(log + ': time: {0:.2f}s'.format(end-start))

#Program reference start timestamp
# start = time.time()
# printTimeStamp(start, "Detection Time")

def cloud_api_predict(headers, payload):
    # send POST request to url
    return requests.request("POST", url, headers=headers, data=payload).text

def get_prediction_data(predictions_json):
    # Parse inference results received from the API call
    res = json.loads(predictions_json) # convert json string to Python dict for parsing
    # opacity=150
    boxes=res[0]["pred_boxes"]
    panoptic_seg=np.array(res[0]["panoptic_seg"])
    instance_list=res[0]["instance_list"]
    return boxes, panoptic_seg, instance_list

# Define Panoptic segmentation visualizer function
def pan_seg_visualizer(panoptic_seg, instance_list, metadata, boxes, opacity):
    stuff_classes=metadata["stuff_classes"]
    stuff_colors=metadata["stuff_colors"]
    thing_classes=metadata["thing_classes"]
    thing_colors=metadata["thing_colors"]
    rgba = np.zeros([480,640,4], dtype = np.uint8)
    rgba[:, :] = [255, 255, 255, 0]
    font = ImageFont.truetype('./fonts/Ubuntu-Bold.ttf', 8)
    stuff_array=[]

    for seg_info in instance_list:
        maskX=(panoptic_seg==seg_info["id"])
        binary_mask=np.array((maskX == True),  dtype=int)
        if (seg_info["isthing"]):
            color=thing_colors[seg_info["category_id"]]
            rgba[(binary_mask == 1), :] = np.append(color, opacity)
        else: # isStuff
            name=stuff_classes[seg_info['category_id']]
#             if (name=="road"): # filter stuff to display
            color=stuff_colors[seg_info["category_id"]]
            x,y = np.argwhere(binary_mask == 1).mean(axis=0)
            stuff_array.append((name,x,y))
            rgba[(binary_mask == 1), :] = np.append(color, opacity)

    maskXImg = Image.fromarray(np.asarray(rgba),mode='RGBA')
    draw = ImageDraw.Draw(maskXImg)
#     Draw boxes and instance labels
    for seg_info, box in zip(instance_list, boxes):
        if(seg_info["isthing"]): #only process things not stuff
            #   text = f"{thing_classes[seg_info['category_id']]} {.85:.0%}" # this only works with python 3.6 and above
            text = "{0} {1:.0%}".format(thing_classes[seg_info['category_id']], seg_info['score'])
            txtSize = font.getsize(text)
            width_text, height_text = txtSize[0], txtSize[1]
            draw.rectangle([(box[0], box[1]-height_text), (box[0] + width_text, box[1])], fill=(0,0,0))#text background rectangle
            draw.text((box[0], box[1]-height_text), text, fill=(255, 255, 255)) # draw text on instance
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=(0,255,0))#blue rectangle

    for stuff in stuff_array:       
        x1=stuff[1]
        y1=stuff[2]
        text=stuff[0]
        txtSize = font.getsize(text)
        width_text, height_text = txtSize[0], txtSize[1]
        draw.rectangle([(y1, x1), (y1 + width_text, x1+height_text)], fill=(0,0,0))#text background rectangle
        draw.text((y1, x1), text, fill=(255, 255, 255)) # draw text on instance

    return maskXImg