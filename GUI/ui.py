from tkinter.filedialog import *
import random
from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk
from tkinter.font import Font

class WinGUI(Tk):
    def __init__(self):
        super().__init__()
        self.__win()
        self.pic_path = ""
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
        self.pyt = None
        self.locate = None
        self.color = None
        self.result = ""
        font = Font(family="Arial", size=24, weight="bold")
        self.tk_label_label_name.configure(font=font)
        font = Font(family="STSong", size=15, weight="bold")
        self.tk_label_label_name_command.configure(font=font)
        self.tk_label_label_name_locate.configure(font = font)
        self.tk_label_label_name_result.configure(font = font)
        font = Font(family="SimHei", size=30, weight="bold")
        self.tk_label_label_result.configure(font = font)
    def __win(self):
        self.title("车牌识别系统")
        # 设置窗口大小、居中
        width = 1000
        height = 700
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)
        
        self.resizable(width=False, height=False)
        
    def scrollbar_autohide(self,vbar, hbar, widget):
        """自动隐藏滚动条"""
        def show():
            if vbar: vbar.lift(widget)
            if hbar: hbar.lift(widget)
        def hide():
            if vbar: vbar.lower(widget)
            if hbar: hbar.lower(widget)
        hide()
        widget.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Leave>", lambda e: hide())
        if hbar: hbar.bind("<Enter>", lambda e: show())
        if hbar: hbar.bind("<Leave>", lambda e: hide())
        widget.bind("<Leave>", lambda e: hide())
    
    def v_scrollbar(self,vbar, widget, x, y, w, h, pw, ph):
        widget.configure(yscrollcommand=vbar.set)
        vbar.config(command=widget.yview)
        vbar.place(relx=(w + x) / pw, rely=y / ph, relheight=h / ph, anchor='ne')
    def h_scrollbar(self,hbar, widget, x, y, w, h, pw, ph):
        widget.configure(xscrollcommand=hbar.set)
        hbar.config(command=widget.xview)
        hbar.place(relx=x / pw, rely=(y + h) / ph, relwidth=w / pw, anchor='sw')
    def create_bar(self,master, widget,is_vbar,is_hbar, x, y, w, h, pw, ph):
        vbar, hbar = None, None
        if is_vbar:
            vbar = Scrollbar(master)
            self.v_scrollbar(vbar, widget, x, y, w, h, pw, ph)
        if is_hbar:
            hbar = Scrollbar(master, orient="horizontal")
            self.h_scrollbar(hbar, widget, x, y, w, h, pw, ph)
        self.scrollbar_autohide(vbar, hbar, widget)
    def connection(self, path):
        ##### to be changed #####
        # 返回路径, 识别结果, 颜色（RGB排列）
        return 'tst.jpg', '苏 CQ123222', '#FF0000'
    def work(self):
        tmp, self.result, self.color = self.connection(self.pic_path)
        image = Image.open(tmp)
        # 获取Label的尺寸
        label_width = self.tk_label_label_locate.winfo_width()
        label_height = self.tk_label_label_locate.winfo_height()
        # 计算图片的缩放比例
        image_width, image_height = image.size
        width_ratio = label_width / image_width
        height_ratio = label_height / image_height
        scale_ratio = min(width_ratio, height_ratio)
        # 缩放图片
        new_width = int(image_width * scale_ratio)
        new_height = int(image_height * scale_ratio)
        print(new_width, new_height)    

        image = image.resize((new_width, new_height), Image.LANCZOS)
        # 将缩放后的图片转换为PhotoImage对象
        self.locate = ImageTk.PhotoImage(image)
        self.tk_label_label_result.configure(text=self.result, background=self.color)
        self.tk_label_label_locate.configure(image=self.locate)

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
    def __tk_frame_frame_origin(self,parent):
        frame = Frame(parent, borderwidth=5, relief='ridge')
        frame.place(x=35, y=175, width=690, height=455)
        return frame    
    def __tk_label_label_origin(self,parent):
        label = Label(parent,anchor="center", )
        label.place(width=680, height=445)
        return label
    def __tk_label_label_name(self,parent):
        label = Label(parent,text="车牌识别系统",anchor="center", )
        label.place(x=315, y=65, width=370, height=74)
        return label
    def __tk_frame_frame_command(self,parent):
        frame = Frame(parent, borderwidth=5, relief='ridge')
        frame.place(x=740, y=175, width=242, height=144)
        return frame
    def __tk_label_label_name_command(self,parent):
        label = Label(parent,text="命令：",anchor="w", )
        label.place(x=0, y=7, width=77, height=30)
        return label
    def __tk_button_button_choose(self,parent):
        btn = Button(parent, command=self.choose_pic, text="选择图片", takefocus=False)
        btn.place(x=46, y=48, width=150, height=30)
        return btn
    def __tk_button_button_apply(self,parent):
        btn = Button(parent, command=self.work, text="识别图片", takefocus=False,)
        btn.place(x=46, y=93, width=150, height=30)
        return btn
    def __tk_frame_frame_locate(self,parent):
        frame = Frame(parent, borderwidth=5, relief='ridge')
        frame.place(x=740, y=372, width=242, height=100)
        return frame
    def __tk_label_label_locate(self,parent):
        label = Label(parent,anchor="center", )
        label.place(x=0, y=0, width=235, height=93)
        return label
    def __tk_frame_frame_result(self,parent):
        frame = Frame(parent, borderwidth=5, relief='ridge')
        frame.place(x=740, y=530, width=242, height=100)
        return frame
    def __tk_label_label_result(self,parent):
        label = Label(parent,anchor="center")
        label.place(x=0, y=0, width=235, height=93)
        return label
    def __tk_label_label_name_locate(self,parent):
        label = Label(parent,text="车牌定位：",anchor="center", )
        label.place(x=740, y=332, width=100, height=30)
        return label
    def __tk_label_label_name_result(self,parent):
        label = Label(parent,text="预测结果：",anchor="center", )
        label.place(x=740, y=490, width=100, height=30)
        return label


class Win(WinGUI):
    def __init__(self, controller):
        self.ctl = controller
        super().__init__()
        self.__event_bind()
        self.__style_config()
        self.ctl.init(self)
    def __event_bind(self):
        pass
    def __style_config(self):
        pass
if __name__ == "__main__":
    win = WinGUI()
    win.mainloop()