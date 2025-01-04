import yolov5
import cv2
import os

def traverse_images(directory):
    """
    遍历目录中的所有图片文件，包括子文件夹
    :param directory: 要遍历的目录路径
    """
    # 定义支持的图片扩展名
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
    image_paths = []

    # 遍历目录及其子目录
    for root, _, files in os.walk(directory):
        for file in files:
            # 获取文件的扩展名并检查是否为有效图片
            if os.path.splitext(file)[-1].lower() in valid_extensions:
                # 拼接完整路径
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    
    return image_paths

# load model
model = yolov5.load('keremberke/yolov5n-license-plate')
  
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# set image
image_paths = traverse_images("test_images")
for image_path in image_paths:
    img = cv2.imread(image_path)

    # perform inference
    results = model(img, size=640)

    # inference with test time augmentation
    results = model(img, augment=True)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # show detection bounding boxes on image
    results.show()

    # save results into "results/" folder
    # results.save(save_dir='results/')
