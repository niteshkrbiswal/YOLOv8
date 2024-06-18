from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results=model.train(data=r"D:\Data science\2. ML\CV Projects\automatic-number-plate-recognition-python-yolov8-main\Nitesh\config.yaml", epochs=1)  # train the model
