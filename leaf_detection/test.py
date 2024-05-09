from ultralytics import YOLO

model = YOLO("new_leaf.pt")

results = model.predict(source="0", show=True)


