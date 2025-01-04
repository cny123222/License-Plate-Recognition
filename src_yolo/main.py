import cv2
from extract import extract_plate_yolo
from split import split_char
from utils import cv_show, traverse_images
from recognition import recognize_characters

def main(image_path):
    origin_image = cv2.imread(image_path)
    plate_image, plate_color = extract_plate_yolo(origin_image)

    print(f"车牌颜色: {plate_color}")
    cv_show("plate", plate_image)

    # char_images = split_char(plate_image)

    # characters = recognize_characters(char_images)
    # print(characters)


if __name__ == "__main__":
    image_paths = traverse_images("test_images")
    for image_path in image_paths:
        main(image_path)
    # main("test_images/002.png")