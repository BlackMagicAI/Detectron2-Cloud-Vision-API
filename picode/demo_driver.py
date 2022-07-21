import io
# importing  all the
# functions defined in detectron2_cloud_vision_api_client.py
from detectron2_cloud_vision_api_client import *


### Run command: python3 demo_driver.py

###
#  Call function
###

# Read image into memory
with open("../images/city-scene.jpg", 'rb') as f:
    payload = f.read()
    
print(type(payload))

headers = {
  'Accept': 'image/jpeg',
  'Content-Type': 'image/jpeg'
}

image_src = Image.open(io.BytesIO(payload)).convert('RGBA')

# Load Model metadata needed to interpret the inference results
# Opening JSON file
with open('./files/pan_metadata.json') as json_file:
    metadata = json.load(json_file)

#Program reference start timestamp
start = time.time()

predictions=cloud_api_predict(headers, payload)

printTimeStamp(start, "Detection Time")

boxes, panoptic_seg, instance_list=get_prediction_data(predictions)
mask=pan_seg_visualizer(panoptic_seg, instance_list, metadata, boxes, 150)

# Overlay Panoptic Segmentation mask over original image
new_image=Image.alpha_composite(image_src, mask)
new_image.show() #display final annotated image
# Or save to file
# new_image.save("output.jpg")

printTimeStamp(start, "Execution End Time")