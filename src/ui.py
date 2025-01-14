from tkinter.filedialog import *
from tkinter import *
# from tkinter.ttk import *
from PIL import Image, ImageTk
from tkinter.font import Font
from extract import extract_plate
from split import split_char
from recognition import recognize_characters
import cv2

COLOR_MAP = {
    "Blue": "#0000FF",
    "Yellow": "#FFFF00",
    "Green": "#008000",
}

class WinGUI(Tk):
    # 初始化窗口 设置各个部件的位置 大小 颜色 字体 对应事件
    # label用于显示图片&文字
    # button用于响应用户的点击 运行函数
    # frame用于放置部件 是label和button的载体
    def __init__(self):
        super().__init__()
        self.__win()
        self.pyt = None # 原始图片 类型为ImageTk.PhotoImage
        self.locate = None # 车牌定位图 类型为ImageTk.PhotoImage
        self.color = "" # 车牌颜色 类型为str
        self.result = "" # 识别结果 类型为str
        self.pic_path = "" # 原图路径 类型为str
        self.tk_frame_frame_origin = self.__tk_frame_frame_origin(self)
        self.tk_label_label_origin = self.__tk_label_label_origin(self.tk_frame_frame_origin)
        self.tk_label_label_name = self.__tk_label_label_name(self)
        self.tk_frame_frame_command = self.__tk_frame_frame_command(self)
        self.tk_label_label_name_command = self.__tk_label_label_name_command( self.tk_frame_frame_command) 
        self.tk_button_button_choose = self.__tk_button_button_choose( self.tk_frame_frame_command) 
        self.tk_button_button_apply = self.__tk_button_button_apply( self.tk_frame_frame_command) 
        self.tk_frame_frame_locate = self.__tk_frame_frame_locate(self)
        self.tk_label_label_locate = self.__tk_label_label_locate( self.tk_frame_frame_locate) 
        self.tk_frame_frame_result = self.__tk_frame_frame_result(self)
        self.tk_label_label_result = self.__tk_label_label_result( self.tk_frame_frame_result) 
        self.tk_label_label_name_locate = self.__tk_label_label_name_locate(self)
        self.tk_label_label_name_result = self.__tk_label_label_name_result(self)

        # 设置字体和字号
        font = Font(family="Arial", size=24, weight="bold")
        self.tk_label_label_name.configure(font=font)
        font = Font(family="STSong", size=15, weight="bold")
        self.tk_label_label_name_command.configure(font=font)
        self.tk_label_label_name_locate.configure(font = font)
        self.tk_label_label_name_result.configure(font = font)
        font = Font(family="SimHei", size=30, weight="bold")
        self.tk_label_label_result.configure(font = font)

    # 设置窗口大小 标题
    def __win(self):
        self.title("车牌识别系统")
        width = 1000
        height = 700
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)
        
        self.resizable(width=False, height=False)

    # 连接识别函数
    def connection(self, image_path):
        """
        调用主程序完成车牌定位、字符分割与识别
        返回定位图路径、识别结果、车牌颜色
        """
        origin_image = cv2.imread(image_path)
        plate_image, plate_color = extract_plate(origin_image)
        char_images = split_char(plate_image)
        characters = recognize_characters(char_images)
        ### upd 2025/1/14 added · between '苏C' and 'Q123222'
        new_characters = characters[0:2]
        new_characters += '·'
        new_characters += characters[2:]
        characters = new_characters

        # 将颜色字符串转换为颜色编码
        color_code = COLOR_MAP.get(plate_color, "#FFFFFF")

        # 保存车牌定位图临时路径
        cv2.imwrite("tmp/temp_plate.jpg", plate_image)
        return "tmp/temp_plate.jpg", characters, color_code
    
    # “识别图片”按键对应的函数 作用是处理图片得到车牌并显示定位和结果
    def work(self):
        """
        点击识别图片按钮后的处理逻辑
        """
        if not self.pic_path:
            print("未选择图片！")
            return

        # 调用识别逻辑
        lp_img, self.result, self.color = self.connection(self.pic_path)

        # 更新车牌定位图
        image = Image.open(lp_img)
        label_width = self.tk_label_label_locate.winfo_width()
        label_height = self.tk_label_label_locate.winfo_height()
        image_width, image_height = image.size
        scale_ratio = min(label_width / image_width, label_height / image_height)
        new_width = int(image_width * scale_ratio)
        new_height = int(image_height * scale_ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)
        self.locate = ImageTk.PhotoImage(image)
        self.tk_label_label_locate.configure(image=self.locate)

        # 更新识别结果
        self.tk_label_label_result.configure(
            text=self.result,
            background=self.color,  # 设置背景为车牌颜色
            foreground="white"     # 设置字体为白色
        )

    # “选择图片”按键对应的函数 作用是选择图片并显示原图
    def choose_pic(self):
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg"),("png图片", "*.png")])
        if self.pic_path:
            image = Image.open(self.pic_path)
            # 获取Label的尺寸
            label_width = self.tk_label_label_origin.winfo_width()
            label_height = self.tk_label_label_origin.winfo_height()
            # 计算图片的缩放比例
            image_width, image_height = image.size
            width_ratio = label_width / image_width
            height_ratio = label_height / image_height
            scale_ratio = min(width_ratio, height_ratio)
            # 缩放图片
            new_width = int(image_width * scale_ratio)
            new_height = int(image_height * scale_ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            # 将缩放后的图片转换为PhotoImage对象
            self.pyt = ImageTk.PhotoImage(image)
            # 配置Label的image属性
            self.tk_label_label_origin.configure(image=self.pyt)
    # 初始化装载原图的frame
    def __tk_frame_frame_origin(self,parent):
        frame = Frame(parent, borderwidth=5, relief='ridge')
        frame.place(x=35, y=175, width=690, height=455)
        return frame    
    # 初始化显示原图的label
    def __tk_label_label_origin(self,parent):
        label = Label(parent,anchor="center", )
        label.place(width=680, height=445)
        return label
    # 初始化label用于显示标题
    def __tk_label_label_name(self,parent):
        label = Label(parent,text="车牌识别系统",anchor="center", )
        label.place(x=315, y=65, width=370, height=74)
        return label
    # 初始化frame用于装载命令栏
    def __tk_frame_frame_command(self,parent):
        frame = Frame(parent, borderwidth=5, relief='ridge')
        frame.place(x=740, y=175, width=242, height=144)
        return frame
    # 初始化label用于显示文字 “命令：”
    def __tk_label_label_name_command(self,parent):
        label = Label(parent,text="命令：",anchor="w", )
        label.place(x=0, y=7, width=77, height=30)
        return label
    # 初始化button用于选择图片
    def __tk_button_button_choose(self,parent):
        btn = Button(parent, command=self.choose_pic, text="选择图片", takefocus=False)
        btn.place(x=46, y=48, width=150, height=30)
        return btn
    # 初始化button用于识别图片
    def __tk_button_button_apply(self,parent):
        btn = Button(parent, command=self.work, text="识别图片", takefocus=False,)
        btn.place(x=46, y=93, width=150, height=30)
        return btn
    # 初始化frame用于装载车牌定位图
    def __tk_frame_frame_locate(self,parent):
        frame = Frame(parent, borderwidth=5, relief='ridge')
        # frame.place(x=740, y=372, width=242, height=100)
        frame.place(x=740, y=372, width=242, height=84) #upd 2025/1/14 adjusted the ratio
        return frame
    # 初始化label用于显示车牌定位图
    def __tk_label_label_locate(self,parent):
        label = Label(parent,anchor="center", )
        # label.place(x=0, y=0, width=235, height=93)
        label.place(x=0, y=0, width=235, height=77) #upd 2025/1/14 adjusted the ratio
        return label
    # 初始化frame用于装载车牌识别结果
    def __tk_frame_frame_result(self,parent):
        frame = Frame(parent, borderwidth=5, relief='ridge')
        # frame.place(x=740, y=530, width=242, height=100)
        frame.place(x=740, y=530, width=242, height=84) #upd 2025/1/14 adjusted the ratio
        return frame
    # 初始化label用于显示车牌识别结果
    def __tk_label_label_result(self,parent):
        label = Label(parent,anchor="center")
        # label.place(x=0, y=0, width=235, height=93)
        label.place(x=0, y=0, width=235, height=77) #upd 2025/1/14 adjusted the ratio
        return label
    # 初始化label用于显示文字 “车牌定位：”
    def __tk_label_label_name_locate(self,parent):
        label = Label(parent,text="车牌定位：",anchor="center", )
        label.place(x=740, y=332, width=100, height=30)
        return label
    # 初始化label用于显示文字 “预测结果：”
    def __tk_label_label_name_result(self,parent):
        label = Label(parent,text="预测结果：",anchor="center", )
        label.place(x=740, y=490, width=100, height=30)
        return label

if __name__ == "__main__":
    win = WinGUI() # 创建窗口
    win.mainloop() # 开始循环（等待用户操作）
