import cv2 as cv
import numpy as np
import os
from typing import Optional
from PIL import Image
from mss import mss
import pyautogui
from ahk import AHK
from pynput import keyboard
import time
import sys
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.block import SPPF
import ultralytics.nn as unn
import ultralytics.utils as uutil
from ultralytics import YOLO
import torch.nn as nn
import torch.serialization
import json
Controls = []
Gdata = ""
try:
    with open('config.json', 'r') as file:
        data = json.load(file)
    
    # Now 'my_data' is a Python dictionary
    Controls = data["Controls"]
    Gdata=data
    print("Config loaded")
except FileNotFoundError:
    print("Error: The file 'data.json' was not found.")
except json.JSONDecodeError:
    print("Error: Could not decode JSON from the file. Check for invalid JSON format.")
torch.serialization.add_safe_globals([DetectionModel])
torch.serialization.add_safe_globals([nn.Sequential])
torch.serialization.add_safe_globals([Conv])
torch.serialization.add_safe_globals([nn.Conv2d])
torch.serialization.add_safe_globals([nn.BatchNorm2d])
torch.serialization.add_safe_globals([nn.modules.activation.SiLU])
torch.serialization.add_safe_globals([C2f])
torch.serialization.add_safe_globals([nn.modules.ModuleList])
torch.serialization.add_safe_globals([Bottleneck])
torch.serialization.add_safe_globals([SPPF])
torch.serialization.add_safe_globals([nn.modules.pooling.MaxPool2d])
torch.serialization.add_safe_globals([nn.modules.upsampling.Upsample])
torch.serialization.add_safe_globals([unn.modules.conv.Concat])
torch.serialization.add_safe_globals([unn.modules.head.Detect])
torch.serialization.add_safe_globals([unn.modules.block.DFL])
torch.serialization.add_safe_globals([uutil.loss.v8DetectionLoss])
torch.serialization.add_safe_globals([nn.modules.loss.BCEWithLogitsLoss])
torch.serialization.add_safe_globals([uutil.tal.TaskAlignedAssigner])
torch.serialization.add_safe_globals([uutil.loss.BboxLoss])
torch.serialization.add_safe_globals([uutil.IterableSimpleNamespace])

print("running test image check...")
model = YOLO(Gdata["Model"])
model.predict("./images/vienta.jpg",conf=0.6)

ahk = AHK(executable_path=Gdata["AHK"])
win = ahk.find_window(title=Gdata["GameTitle"])
def FindImage(imagepath, matchvalue):
    w,h = pyautogui.size()
    monitor = {"top":0, "left":0, "width":w, "height":h}
    FoundOne = False
    with mss() as sct:
        screen = sct.grab(monitor)
        screen = np.array(screen)
        screen = cv.cvtColor(screen,cv.COLOR_RGB2BGR)

        imageToFind = cv.imread(imagepath)
        imageToFind = cv.cvtColor(imageToFind,cv.COLOR_RGB2BGR)

        result = cv.matchTemplate(imageToFind, screen, cv.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        if (max_val > matchvalue):
            #print(f"Found image {max_val}")
            FoundOne = True
            TopLeft = max_loc
            ImageHalfWidth = imageToFind.shape[1]/2
            ImageHalfHeight = imageToFind.shape[0]/2
            Center = (TopLeft[0]+ImageHalfWidth,TopLeft[1]+ImageHalfHeight)
        else:
            pass
    return [FoundOne,max_val]
def on_press(key):
    try:
        if (key == keyboard.Key.backspace):
            os._exit(0)
    except:
        print("err")
listener = keyboard.Listener(
    on_press=on_press)
listener.start()
OneTime = 0
OriginX = 0
NoDetectionCounter = 0
while True:
    OneTime+=1
    if OneTime == 1:
        ahk.key_press(Controls["interact"])
    FoundFirstImage = FindImage("./images/fish.jpg", 0.67)
    if FoundFirstImage[0] == True:
        ahk.key_press(Controls["reel_in"])
        with mss() as sct:
            w,h = pyautogui.size()
            monitor = {"top":0, "left":0, "width":w, "height":h}
            while True:
                screen = sct.grab(monitor)
                screen = np.array(screen)
                screen = cv.cvtColor(screen, cv.COLOR_BGRA2BGR)

                results = model.predict(screen,stream=True,conf=0.6)
                FindEmptyStam = FindImage("./images/empty_stam.jpg",0.8)
                if FindEmptyStam[0] == True:
                    ahk.key_up(Controls["right"])
                    ahk.key_up(Controls["left"])
                    time.sleep(0.26)
                for r in results:
                    boxes=r.boxes
                    if len(boxes) > 0:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy.int().tolist()[0]  # Top-left (x1, y1), Bottom-right (x2, y2)
                            width= x2-x1 #bigger x gets minus with smaller x to get width
                            CenterX = x1 + width/2
                            if OriginX == 0:
                                OriginX = CenterX
                            if CenterX > OriginX:
                                ahk.key_up(Controls["right"])
                                ahk.key_down(Controls["left"])
                            elif CenterX < OriginX:
                                ahk.key_up(Controls["left"])
                                ahk.key_down(Controls["right"])
                    else:
                        NoDetectionCounter += 1

                if NoDetectionCounter == 10:
                    NoDetectionCounter = 0
                    OneTime = 0
                    OriginX = 0
                    ahk.key_up(Controls["right"])
                    ahk.key_up(Controls["left"])
                    time.sleep(6)
                    break
                        

            
