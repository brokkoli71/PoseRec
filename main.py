import tflite_runtime.interpreter as tflite
import numpy as np
import sys
import cv2
import math
import time
import picamera

lables = ["nose","leftEye","rightEye","leftEar","rightEar","leftShoulder","rightShoulder","leftElbow","rightElbow","leftWrist","rightWrist","leftHip","rightHip","leftKnee","rightKnee","leftAnkle","rightAnkle"]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def resize_imgfile(img, scale_factor=1.0, output_stride=16):
    img = cv2.resize(img, (257, 257))
    return _process_input(img, scale_factor, output_stride)

def _process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.reshape(1, target_height, target_width, 3)
    return input_img, source_img, scale

def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height

with picamera.PiCamera() as camera:
    camera.resolution = (720, 720)
    camera.vflip = True
    camera.capture("image/03.jpg")
print("captured a picture")
start = time.time()
np.set_printoptions(threshold=sys.maxsize, precision=5, floatmode="fixed")

img = cv2.imread("image/03.jpg") #timeconsuming: 0.09s
print(start-time.time())
input_image, draw_image, output_scale = resize_imgfile(img, scale_factor=1, output_stride=16)

interpreter = tflite.Interpreter(model_path='posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# set input
input_shape = input_details[0]['shape']
input_data = input_image
interpreter.set_tensor(input_details[0]['index'], input_data)
print(start-time.time())
interpreter.invoke() #obviously timeconsuming: 0.12s
print(start-time.time())
# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
heatmaps_result = interpreter.get_tensor(output_details[0]['index'])
offsets_result = interpreter.get_tensor(output_details[1]['index'])
displacement_fwd_result = interpreter.get_tensor(output_details[2]['index'])
displacement_bwd_result = interpreter.get_tensor(output_details[3]['index'])

height, width, numKeypoints = heatmaps_result.shape[1:]

#find locations of where the keypoints are most likely to be.
keypoints = []
for i in range(height):
    for j in range(width):
        for k in range(numKeypoints):
            if(heatmaps_result[0][i][j][k]>0.9):
                keypoints.append([i, j, k])

imgHeight, imgWidth = 257, 257



points = []

for keypoint in keypoints:
    #richtig?
    posY = keypoint[0]
    posX = keypoint[1]
    lableID = keypoint[2]
    y = posY / (height-1) * imgHeight + offsets_result[0][posY][posX][lableID]
    x = posX / (width-1) * imgWidth + offsets_result[0][posY][posX][lableID+numKeypoints]
    confidence = sigmoid(heatmaps_result[0][posY][posX][lableID])
    lable = lables[lableID]
    points.append([int(x), int(y), confidence, lable])

print(start-time.time())
scale = img.shape[0]/257
img = cv2.resize(img, (int(257*scale), int(257*scale)))

for point in points:
    img = cv2.circle(img, (int(point[0]*scale), int(point[1]*scale)), int(scale*2), (0, 0, 255), -1)

#cv2.imwrite('output.png',img)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


