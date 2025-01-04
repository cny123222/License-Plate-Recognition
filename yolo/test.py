from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # 测试加载 YOLOv8 预训练模型
model.predict(source='path_to_image.jpg', show=True)