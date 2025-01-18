import cv2
from extract import extract_plate_yolo
from split_char import split_char
from utils import cv_show, traverse_images
from retina import extract_retina
from color import detect_plate_color
import numpy as np
import os
# from recognition import recognize_characters

def main(image_path):
    origin_image = cv2.imread(image_path)
    plate_image = extract_plate_yolo(origin_image)

    filename = os.path.basename(image_path)
    path = os.path.join("tmp", filename)

    if plate_image is not None:
        cv2.imwrite(path, plate_image)

    if plate_image is None:
        plate_retina = extract_retina(image_path, path)
        if not plate_retina:
            chars = split_char(image_path)
            if(not chars):
                print("没有检测到车牌！！！")
                return chars

        else:
            chars = split_char(path)
    else:
        plate_retina = extract_retina(path, path)
        


    plate_color = detect_plate_color(cv2.imread(path))

    # print(f"车牌颜色: {plate_color}")
    # cv_show("plate", plate_image)

    #cv2.imwrite("plate.jpg",plate_image)
    chars = split_char(path)

    # print(chars)
    if(chars[2] == '.'):
        new_chars = chars[0:2]
        new_chars += '.'
        new_chars += chars[2:]
        chars = new_chars
    if(len(chars) == 9):
        plate_color = "Green"
    return chars
    #print(f"车牌颜色: {plate_color}")
    #print(chars)



    # char_images = split_char(plate_image)

    # characters = recognize_characters(char_images)
    # print(characters)


if __name__ == "__main__":
    # image_paths = traverse_images("test_images")
    # for image_path in image_paths:
    #     main(image_path)
    main("000.jpg")