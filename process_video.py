import os
import boto3
import requests
import urllib
import re
import tqdm
from PIL import Image
from io import BytesIO
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.animation as animation
import cv2
import time
from pathlib import Path
import traceback
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import urllib
import cv2
import tensorflow as tf
import numpy as np
import time
import traceback
import json
import uvicorn
import boto3
import logging

app = FastAPI()

s3_client = boto3.client('s3')

class RequestPayload(BaseModel):
    local_filename: str
    output_filename: str
    lambda_request_id: str


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Process %(process)d] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
    #vggloss
    lvgg=K.mean(K.abs(lossmodel(ytrue)-lossmodel(ypred)))
    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta
    w4=0.1

    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))+w4*lvgg

#keras.backend.normalize_data_format=normalize_data_format
def ssim_loss(y_true, y_pred):
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, 1)) * 0.5, 0, 1)
    #vggloss
    lvgg=K.clip(K.mean(K.abs(lossmodel(y_true)-lossmodel(y_pred))),0,1)
    return l_ssim+tf.keras.losses.MAE(y_true, y_pred)+lvgg

def animate(U,D, savename='test',show=False):
    ims=[]
    fig = plt.figure()
    for i in range(10):
        if i%2==0:
            im = plt.imshow(U, animated=True)
            plt.axis('off')
        else:
            im = plt.imshow(D, animated=True)
            plt.axis('off')
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=10)
    ani.save(savename+'.gif', writer='imagemagick', fps=10)
    if show:
        plt.show()
    return ani

def creategif(img):
  Dp=downfromup.predict(img)[:,:,:].clip(0,1)
  animate(img, Dp,'animatedted')
  return 0

def convertvideo(videoname, modelconverter, outname, size=None):
    vidcap = cv2.VideoCapture(videoname)
    success, image = vidcap.read()
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    if size == None:
        shapeout = (int(np.ceil(image.shape[1] / 32) * 32), int(np.ceil(image.shape[0] / 32) * 32) * 2)
    else:
        shapeout = (size[0] // 32 * 32, size[1] // 32 * 32 * 2)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    startflag = 0
    count = 0
    ctr = 0
    timein = time.time()

    while success:
        try:
            success, image = vidcap.read()
            print(count)
            count += 1
            ctr = ctr + 1

            imageU = tf.image.resize_with_pad(image, shapeout[1] // 2, shapeout[0])
            U = imageU
            CC = np.expand_dims(np.transpose(np.indices((U.shape[0], U.shape[1])) / U.shape[0], (1, 2, 0)), 0)
            D = modelconverter.predict([np.stack([U / 255]), CC])[0, :, :, :].clip(0, 1)
            #  D = D * 2.0  # Increase the depth range by multiplying with a factor
            # D = np.exp(D)  # Apply an exponential scaling function to the depth values
            # D = cv2.bilateralFilter(D, 9, 75, 75)  # Apply bilateral filtering to the depth map

            frame = np.concatenate([U, D * 255], 0).astype(np.uint8)

            if startflag == 0:
                out = cv2.VideoWriter(outname, fourcc, int(fps), (int(frame.shape[1]), int(frame.shape[0])))
                startflag = 1

            out.write(frame)

            if ctr % (fps) == 0:
                print(ctr // (fps), ' s ', time.time() - timein, ' sec per videosecond')
                timein = time.time()

        except:
            print(print(traceback.format_exc()))
            break

    vidcap.release()
    out.release()
    return frame

def convertimage(imagepath, converter):
  img=cv2.imread(imagepath)
  #flip channels from bgr to rgb
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img=np.stack([img])
  shapeout=(int(img.shape[1]//32-0.000001)*32+32,(int(img.shape[2]//32-0.000001)*32+32),3)
  img=img[:,:shapeout[0],:shapeout[1],:shapeout[2]]
  CC=np.expand_dims(np.transpose(np.indices((img.shape[1],img.shape[2]))/img.shape[1],(1,2,0)),0)
  predictedimage=converter.predict([img, CC]).astype(np.uint8)
  return img[0,:,:,:], predictedimage[0,:,:,:].clip(0,255)


# initiate model
linkurl = 'http://aidle.org/js/nns/nonsymmdensenet2_freezed640_0.072_01-0.063.h5'
Ans = urllib.request.urlretrieve(linkurl, 'nueral2d3d.h5')
BilinearUpSampling2D = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function, 'ssim_loss': ssim_loss}
downfromup = tf.keras.models.load_model('nueral2d3d.h5', custom_objects=custom_objects)
downfromup.compile(loss='mse', optimizer='Adam')


def processvideo(local_filename: str, output_filename: str):
    try:
        logging.info(f"start processing {local_filename} to {output_filename}")
        frame = convertvideo(local_filename, downfromup, output_filename, size=(640, 640))
        logging.info('video processed')
    except Exception as e:
        logging.info(str(e))



@app.post("/invocations")
def invocations(payload: RequestPayload):
    directory = "/app"
    for filename in os.listdir(directory):
        logging.info(f"file in {directory}: {filename}")
    lambda_request_id = payload.lambda_request_id
    logging.info(f"lambda_request_id: {lambda_request_id}")
    downloadfile_key = payload.local_filename #asdasd.mp4
    uploadfile_key = payload.output_filename #asdasd-converted.mp4
    input_bucket_name = "2dvideouploadbucket"
    output_bucket_name = "video-s3-poc"

    s3_client.download_file(input_bucket_name, downloadfile_key, downloadfile_key)
    result = processvideo(downloadfile_key, uploadfile_key)
    s3_client.upload_file(uploadfile_key, output_bucket_name, uploadfile_key)


    filepath = os.path.join(directory, downloadfile_key)
    os.remove(filepath)
    logging.info(f"Deleted {filepath}")
    filepath = os.path.join(directory, uploadfile_key)
    os.remove(filepath)
    logging.info(f"Deleted {filepath}")

    return JSONResponse({"output_filename": uploadfile_key})


# Define the /ping endpoint
@app.get("/ping")
async def ping():
    # Return an HTTP status code of 200 and an empty response body
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("process_video:app", host="0.0.0.0", port=8080)