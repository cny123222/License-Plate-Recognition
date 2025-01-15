from rapidocr_onnxruntime import RapidOCR


def split_char(image_path):
    '''
    image_path : 图像地址
    return  result:  车牌汉字
    return len : 车牌字符串长度
    '''
    engine = RapidOCR()
    result, elapse = engine(image_path, use_det=False, use_cls=False, use_rec=True)
    result = result[0][0]
    return result

if __name__ == "__main__":
    image_path = '2.jpg'
    split_char(image_path)