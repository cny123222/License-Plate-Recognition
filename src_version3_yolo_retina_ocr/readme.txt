这是实现车牌识别大作业的py代码，其中，color.py用于检测车牌的颜色，在提取过程中，extract.py先使用yolov5对车牌进行粗提取，框出车牌的大致位置，然后Retina.py进一步对车牌进行定位，并进行相应的矫正，
然后spilt_char.py使用ocr对车牌进行识别。这些功能文件都在main.py和ui.py中得到引用。
在运行代码之前，请保证下载了yolov5（需要外网），rapidocr_onnxruntime, tkinter, PIL等外部库。