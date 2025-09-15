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
from mss import mss
import cv2 as cv
import pyautogui
import numpy as np
from pynput import keyboard
import os

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

model = YOLO('./runs/detect/train2/weights/best.pt')

def on_press(key):
    try:
        if (key == keyboard.Key.backspace):
            os._exit(0)
    except:
        print("err")
listener = keyboard.Listener(
    on_press=on_press)
listener.start()

def FindAiScanImage():
    with mss() as sct:
        while True:
            w,h = pyautogui.size()
            monitor = {"top":0, "left":0, "width":w, "height":h}
            screen = sct.grab(monitor)
            screen = np.array(screen)
            screen = cv.cvtColor(screen, cv.COLOR_BGRA2BGR)

            results = model.predict(screen,stream=True)
            for r in results:
                boxes=r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy.int().tolist()[0]  # Top-left (x1, y1), Bottom-right (x2, y2)
                    # These coordinates represent the pixel positions on the captured screen frame.
                    print(f"Detected object at: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                
                annotated_frame = r.plot() # YOLOv8 provides a convenient plot method
                cv.imshow("YOLOv8 Live Detection", annotated_frame)
                cv.waitKey(1)

FindAiScanImage()