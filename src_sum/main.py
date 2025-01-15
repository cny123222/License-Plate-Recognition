import cv2
from extract import extract_plate_yolo
from split_char import split_char
from utils import cv_show, traverse_images
from retina import extract_retina
# from recognition import recognize_characters

def main(image_path):
    origin_image = cv2.imread(image_path)
    plate_image, plate_color = extract_plate_yolo(origin_image)


    if(plate_image is None):
        plate_retina = extract_retina(origin_image)
        if(plate_retina == False):
            chars = split_char(image_path)
            if(chars == None):
                print("没有检测到车牌！！！")
                return
        else:
            chars = split_char("plate.jpg")
    else:
        plate_retina = extract_retina(plate_image)



    # print(f"车牌颜色: {plate_color}")
    cv_show("plate", plate_image)

    cv2.imwrite("plate.jpg",plate_image)
    chars = split_char("plate.jpg")

    # print(chars)

    if(len(chars) == 9):
        plate_color = "Green"

    print(f"车牌颜色: {plate_color}")
    print(chars)



    # char_images = split_char(plate_image)

    # characters = recognize_characters(char_images)
    # print(characters)


if __name__ == "__main__":
    # image_paths = traverse_images("test_images")
    # for image_path in image_paths:
    #     main(image_path)
    main("2.jpg")