import os
import shutil
from PIL import Image

def rename_and_save_images(input_dir, output_dir):
    """
    提取所有图片文件，重新编号后存入目标文件夹
    :param input_dir: 原始文件夹路径
    :param output_dir: 存放重新编号图片的目标文件夹路径
    """
    # 创建目标文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 定义支持的图片格式
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

    # 初始化编号
    counter = 1

    # 遍历所有文件和子文件夹
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_ext = os.path.splitext(file)[-1].lower()
            if file_ext in valid_extensions:
                # 构造图片完整路径
                file_path = os.path.join(root, file)

                # 构造目标文件路径，使用递增编号
                new_filename = f"{counter:03d}{file_ext}"
                new_file_path = os.path.join(output_dir, new_filename)

                # 打开图片并保存到目标文件夹
                try:
                    img = Image.open(file_path)
                    img.save(new_file_path)
                    print(f"已保存: {new_file_path}")
                    counter += 1
                except Exception as e:
                    print(f"无法处理文件: {file_path}, 错误: {e}")

if __name__ == "__main__":
    input_folder = "dataset/test"
    output_folder = "test_images"
    rename_and_save_images(input_folder, output_folder)