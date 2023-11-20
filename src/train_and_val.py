from ultralytics import YOLO

# 1. auto
model = YOLO('yolov8n-seg.pt')
results = model.train(data='custom-seg.yaml', imgsz=320, batch=8, seed=44)

metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps 

# 2. Adam

model = YOLO('yolov8n-seg.pt')
results = model.train(data='custom-seg.yaml', optimizer='Adam', lr0=0.001, epochs=1000, imgsz=320, batch=8, seed=44)

metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps 

# 3. SGD
model = YOLO('yolov8l-seg.pt')
results = model.train(data='custom-seg.yaml', optimizer='SGD', lr0=0.001, epochs=100, imgsz=320, batch=8, seed=44)

metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps 

# 4. NAdam

model = YOLO('yolov8l-seg.pt')
results = model.train(data='custom-seg.yaml', optimizer='NAdam', lr0=0.001, epochs=100, imgsz=320, batch=8, seed=44)

metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps 
