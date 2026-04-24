from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(unfreeze_bn_and_bias=True, freeze=9)
