from paddleocr import PaddleOCR

# 初始化 OCR，设置自定义字符集
ocr = PaddleOCR(det=True, rec=True, use_angle_cls=True, lang='ch', rec_char_dict_path='./chars.txt')

# 推理
result = ocr.ocr('../Pytorch_Retina_License_Plate/test.png', cls=True)
for line in result[0]:
    print("识别结果:", line[1][0])  # 识别结果
    print("置信度:", line[1][1])  # 置信度