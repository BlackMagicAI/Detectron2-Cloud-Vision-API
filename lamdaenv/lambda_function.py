import os
import json
import boto3
from base64 import b64encode,b64decode
from io     import BytesIO
from PIL import Image

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']

# Ref:
# https://docs.aws.amazon.com/lambda/latest/dg/lambda-deploy-functions.html
# https://docs.aws.amazon.com/lambda/latest/dg/python-package.html#python-package-create-package-no-dependency
def lambda_handler(event, context):
    runtime= boto3.client('runtime.sagemaker')
    # Base64 decode data it came in encoded
    img_string = event.get("body","")
    img_data = b64decode(img_string)

    img_buffer = BytesIO(img_data)
    img_buffer.seek(0)
    image_src = Image.open(img_buffer)
    
    imgByteArr = BytesIO()
    image_src.save(imgByteArr, format=image_src.format)
    imgByteArr = imgByteArr.getvalue()
        
    # Send image via InvokeEndpoint API
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='application/x-image', Body=imgByteArr)
    result = response['Body'].read().decode()
    return result
