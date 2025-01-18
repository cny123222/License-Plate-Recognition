import cv2
from extract import extract_plate
from split import split_char
from utils import plt_show_color, traverse_images
from recognition import recognize_characters

def main(image_path):
    origin_image = cv2.imread(image_path)
    try:
        plate_image, plate_color = extract_plate(origin_image)
    except TypeError:
        return

    print(f"车牌颜色: {plate_color}")
    plt_show_color(plate_image)

    char_images = split_char(plate_image)

    characters = recognize_characters(char_images)
    print(characters)


if __name__ == "__main__":
    # image_paths = traverse_images("test_images")
    # for image_path in image_paths:
    #     main(image_path)
    main("test_images/029.jpg")