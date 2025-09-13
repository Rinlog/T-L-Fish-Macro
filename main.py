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

ahk = AHK(executable_path="C:\\Program Files\\AutoHotkey\\AutoHotKey.exe")
win = ahk.find_window(title="TL 1.365.22.4243")
def get_images_from_folder(folderPath):
    image_list = []
    for filename in os.listdir(folderPath):
        if filename.lower()[str.__len__(filename)-4:] == ".jpg":
            image_path = os.path.join(folderPath,filename)
            image_list.append(image_path)
        else:
            print("the image with the name " +filename+ " is not a jpg, please use jpg only")

    return image_list

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
            print(f"Found image {max_val}")
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
while True:
    OneTime+=1
    if OneTime == 1:
        ahk.key_press("q")
    FoundFirstImage = FindImage("./images/fish.jpg", 0.7)
    Maxval = 0
    Continue = True
    if FoundFirstImage[0] == True:
        ahk.key_press("home")
        time.sleep(1.5)
        while True:
            FoundFourthBar = FindImage("./images/bar 4.jpg", 0.75)
            FoundSecondBar = FindImage("./image/bar_2.jpg",0.8)
            if FoundFourthBar[0] == True:
                if Maxval+1 < FoundFourthBar[1]:
                    Maxval = FoundFourthBar[1]
                    Continue = True
                else:
                    Continue = False
                if Continue == True:
                    ahk.key_up("d")
                    ahk.key_down("a")
                else:
                    ahk.key_up("a")
                    ahk.key_down("d")
                time.sleep(1)
                ahk.key_up("a")
                ahk.key_up("d")
                time.sleep(0.1)
            else:
                OneTime = 0
                break


            
