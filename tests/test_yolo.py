from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(unfreeze_bn_and_bias=True, optimizer="SGD", lrs_per_layer={0: 0.01, 1: 0.001, 2: 0.0001})
