import cv2 as cv
import numpy as np
import os
from typing import Optional
from PIL import Image
from mss import mss
import pyautogui
from ahk import AHK
import time

ahk = AHK(executable_path="C:\\Program Files\\AutoHotkey\\AutoHotKey.exe")
win = ahk.find_window(title="TL 1.365.22.4243")

while True:
    ahk.key_press("home")
    time.sleep(0.05)