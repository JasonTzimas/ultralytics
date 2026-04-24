from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(unfreeze_bn=True, freeze=9, lrs_per_layer={1: 0.01, 2: 0.001})
