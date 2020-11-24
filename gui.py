import sys
import tkinter as tk
import fitCorrelationFunctionForGui

class Frame(tk.Frame):
    #コンストラクタ
    def __init__(self, master=None):
        super().__init__(master)


#ウィンドウ生成
root = tk.Tk()
root.title(u"自己相関関数フィッティング")
root.geometry("400x300")



root.mainloop()