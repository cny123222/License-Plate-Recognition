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
    
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
             "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
    ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

    filtered_results = []

    if len(result) <= 2:
           return False
    
    else:
        if result[0] in provinces:
                filtered_results.append(result[0])
        else:
                print("检测失败")
                return False

        if result[1] in alphabets:
                filtered_results.append(result[1])
        else:
                print("检测失败")
                return False

        for i in result[2:]:
                if i in ads:
                        filtered_results.append(i)
    
    results = ''.join(filtered_results)
 
    return results if 7 <= len(results) <= 8 else False

if __name__ == "__main__":
    image_path = '1.jpg'
    split_char(image_path)