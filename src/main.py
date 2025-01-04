import cv2
from extract import extract_plate
from split import split_char
from utils import plt_show_color
from recognition import recognize_characters

def main(image_path):
    origin_image = cv2.imread(image_path)
    plate_image, plate_color = extract_plate(origin_image)

    # print(f"车牌颜色: {plate_color}")
    # plt_show_color(plate_image)

    char_images = split_char(plate_image)

    characters = recognize_characters(char_images)
    print(characters)


if __name__ == "__main__":
    main("dataset/test/timg1.jpg")
