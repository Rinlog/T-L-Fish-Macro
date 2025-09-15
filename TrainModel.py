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
torch.serialization.add_safe_globals([uutil.IterableSimpleNamespace])

model = YOLO('./runs/detect/train6/weights/best.pt')

model.train(data="My-First-Project-3/data.yaml",epochs=100, imgsz=640)