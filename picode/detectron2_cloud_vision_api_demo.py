# import some common libraries
# Using Python Request library
import requests
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Define Constants
IMAGE_PATH="<INSERT_IMAGE_FILE_PATH_HERE>"
API_INVOKE_URL="<INSERT_API_INVOKE_URL>"

# define variables
url=API_INVOKE_URL

# Read image into memory
with open(IMAGE_PATH, 'rb') as f:
    payload = f.read()
    
headers = {
  'Accept': 'image/jpeg',
  'Content-Type': 'image/jpeg'
}
# send POST request to url
response = requests.request("POST", url, headers=headers, data=payload)

# Load Model metadata needed to interpret the inference results
# Opening JSON file
with open('./files/pan_metadata.json') as json_file:
    metadata = json.load(json_file)

# Parse inference results received from the API call
res = json.loads(response.text) # convert json string to Python dict for parsing
opacity=150
boxes=res[0]["pred_boxes"]
panoptic_seg=np.array(res[0]["panoptic_seg"])
instance_list=res[0]["instance_list"]

# Define Panoptic segmentation visualizer function
def pan_seg_visualizer(predictionsSegs, instance_list, image_src, stuff_classes, stuff_colors, thing_classes, thing_colors, boxes, opacity):
    rgba = np.zeros([480,640,4], dtype = np.uint8)
    rgba[:, :] = [255, 255, 255, 0]
    font = ImageFont.truetype('./fonts/Ubuntu-Bold.ttf', 8)
    print(font.getsize("text"))
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

    new_image=Image.alpha_composite(image_src, maskXImg)
    new_image.show() #display final annotated image
    # Or save to file
    # new_image.save("output.jpg")

# Call function
image_src = Image.open(IMAGE_PATH).convert('RGBA')
pan_seg_visualizer(panoptic_seg, instance_list, image_src, metadata["stuff_classes"], metadata["stuff_colors"], metadata["thing_classes"], metadata["thing_colors"], boxes, opacity)