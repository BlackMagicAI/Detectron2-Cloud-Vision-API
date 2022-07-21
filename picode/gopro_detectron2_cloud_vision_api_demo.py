import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from picamera import PiCamera
from picamera.array import PiRGBArray
import base64
import json
from io import BytesIO
import io
import os
import subprocess
# Using Python Request library
import requests
from detectron2_cloud_vision_api_client import *

#########################################
# File:   gopro_detectron2_cloud_vision_api_demo.py
# GoPro camera Detectron2 Panoptic Segmentation using custom cloud vision API on AWS API Gateway.
# Author: Maurice Tedder
# Date:   July 21, 20212
##Ref:
##  https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python
##  https://picamera.readthedocs.io/en/release-1.13/recipes1.html#overlaying-images-on-the-preview
##  https://picamera.readthedocs.io/en/release-1.13/api_camera.html#module-picamera
##  https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray
##  https://github.com/raspberrypilearning/the-all-seeing-pi/blob/master/code/overlay_functions.py

headers = {
  'Accept': 'image/jpeg',
  'Content-Type': 'image/jpeg'
}

# Load Model metadata needed to interpret the inference results
# Opening JSON file
with open('./files/pan_metadata.json') as json_file:
    metadata = json.load(json_file)

######Start function definitions
# Transform rgb to jpeg.
def rgb_to_jpeg_image(image):
  infile = BytesIO()
  image.save(infile, format="JPEG")
  return infile

#Used to print time lapsed during different parts of code execution.
def printTimeStamp(start, log):
  end = time.time()
  print(log + ': time: {0:.2f}s'.format(end-start))

# Combine original image & bounding box annotation overlays image into one image
# & save to jpg file.
# From Ref: https://github.com/raspberrypilearning/the-all-seeing-pi/blob/master/code/overlay_functions.py
def output_overlay(filepath, output=None, overlay=None):

  # Take an overlay Image
  overlay_img = overlay.convert('RGBA')

  # ...and a captured photo
  output_img = output.convert('RGBA')

  # Combine the two and save the image as output
  new_output = Image.alpha_composite(output_img, overlay_img)
  new_output.save(filepath, "JPEG")

######End function definitions

# Create camera & rawcapture objects
#stream = BytesIO()
camera = PiCamera()
camera.resolution=(640,480)
rawCapture = PiRGBArray(camera, size=camera.resolution)
# print(camera.resolution)
#camera.start_preview(fullscreen=False, window=(0,0,size[0],size[1])) #show preview in custom size. Used for debugging.
camera.start_preview(fullscreen=True)

pad = None

for frame in camera.capture_continuous(rawCapture, format='rgb', use_video_port=True):
  
  image = Image.fromarray(frame.array)
  infile=rgb_to_jpeg_image(image)

  #Program reference start timestamp
  start = time.time()

  predictions=cloud_api_predict(headers, infile.getvalue())

  printTimeStamp(start, "Detection Time")

  boxes, panoptic_seg, instance_list=get_prediction_data(predictions)
  mask=pan_seg_visualizer(panoptic_seg, instance_list, metadata, boxes, 150)

  #Remove previous overlays
  for o in camera.overlays:
    camera.remove_overlay(o)

  o = camera.add_overlay(mask.tobytes(), alpha = 255, layer = 3, size=mask.size)

  rawCapture.truncate(0)
  #break #uncomment to end loop and output image to jpg file

# Combine original image & bounding box annotation overlays image into one image
# & save to jpg file.
# output_overlay("out/goproyolo.jpg", image, pad)
printTimeStamp(start, "Execution End Time")

camera.stop_preview()
camera.close()
