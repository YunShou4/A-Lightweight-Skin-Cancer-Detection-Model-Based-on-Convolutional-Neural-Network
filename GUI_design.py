import tkinter as tk
import tkinter.messagebox
from tkinter.filedialog import askopenfilename
from PIL import Image
from PIL import ImageTk
from predict import predict
from tensorflow.keras.models import load_model


class Frame(tk.Tk):

    def __init__(self):
        super().__init__()
        self.filename = None
        self.title("CANCER RADAR")
        self.geometry("700x500")
        # self.identifier = Identifier()
        self.init()


    def init(self):
        # 文件路径
        self.e1 = tk.Entry(self)
        self.e1.place(x=80,y=30)
        self.e2 = tk.Entry(self)
        self.e2.place(x=80,y=110)
        self.entry_text = tk.StringVar()
        self.e3 = tk.Entry(self,textvariable=self.entry_text,width=30)
        self.e3.place(x=80, y=190)
        # 创建按钮
        self.b1 = tk.Button(self, text="choose picture", command=self.getpath)
        self.b1.place(x=230,y=25)
        self.b2 = tk.Button(self, text="start detecting", command=self.do_identify)
        self.b2.place(x=100,y=70)
        self.b3 = tk.Button(self, text = "send", command=self.answer)
        self.b3.place(x=250, y=155)
        # 创建标签
        self.l2 = tk.Label(self, text="chatbox")
        self.l2.pack()
        self.l2.place(x=150,y=160)

        self.label_text = tk.StringVar()
        self.label_text.set("----")
        self.l3 = tk.Label(self,textvariable=self.label_text)
        self.l3.pack()
        self.l3.place(x=100,y=230)


    def getpath(self):
        self.filename = askopenfilename()
        print(self.filename)
        self.e1.delete(0, tk.END)
        self.e1.insert(0, self.filename)
        self.img = Image.open(self.filename)
        self.img = self.img.resize([200, 200], Image.ANTIALIAS)
        panel = tk.Label(master=self)
        panel.photo = ImageTk.PhotoImage(self.img)
        self.l1 = tk.Label(self, image=panel.photo)
        self.l1.place(x=400, y=20)


    def answer(self):
        print(self.entry_text.get())

        self.tinydict = {'name': 'My name is cancer radar',
                         'skin cancer': 'Skin cancer — the abnormal growth of skin cells\n — most often develops on skin exposed to the sun.\n But this common form of cancer can also occur on areas of\n your skin not ordinarily exposed to sunlight.',
                         'job': 'I can detect 7 kinds of skin cancers which are Melanocytic\n nevi, Melanoma, Benign keratosis-like lesions, \nBasal cell carcinoma, Actinic keratoses, Vascular lesions and \nDermatofibroma',
                         'protect': 'You can reduce your risk of skin cancer by limiting\n or avoiding exposure to ultraviolet (UV) radiation.\n Checking your skin for suspicious changes can help detect\n skin cancer at its earliest stages. Early detection of\n skin cancer gives you the greatest chance for \nsuccessful skin cancer treatment.',
                         'Melanocytic nevi' : 'Melanocytic nevi are benign proliferations of melanocytes\n located at different skin levels. Pigmentation in\n different shades of brown can range from a brown-yellowish\n to brown-blackish color; sometimes-as in mature\n cellular nevi-they can be skin-colored.',
                         'Melanoma':'Melanoma, the most serious type of skin cancer, develops in\n the cells (melanocytes) that produce melanin — the\n pigment that gives your skin its color. ',
                         'Benign keratosis-like lesions':'Benign Keratosis is also called Seborrheic Keratoses (SK) are\n the most common skin lesion. It tend to be most \ncommon on sun-exposed areas in older patients.',
                         'Basal cell carcinoma': 'A type of skin cancer which develops in basal cells, a type \nof cell within the skin t produces new skin cells.',
                         'Actinic keratoses':'A condition which causes scaly patches on the skin from exposure\n to the sun over the years. It is commonly found on \nface, lips, ears, neck, back of the hand and forearms.',
                         'Vascular lesions':'Vascular lesions are relatively common abnormalities of the skin\n and underlying tissues, more commonly known as birthmarks.',
                         'Dermatofibroma' :'A dermatofibroma is a common overgrowth of the fibrous tissue\n situated in the dermis (the deeper of the two main layers\n of the skin).'}
        self.where = self.entry_text.get().find('name')
        if self.entry_text.get().find('name') != -1:
            self.label_text.set(self.tinydict['name'])
        elif self.entry_text.get().find('skin cancer') != -1:
            self.label_text.set(self.tinydict['skin cancer'])
        elif self.entry_text.get().find('protect') != -1:
            self.label_text.set(self.tinydict['protect'])
        elif self.entry_text.get().find('job') != -1:
            self.label_text.set(self.tinydict['job'])
        elif self.entry_text.get().find('Melanocytic nevi') != -1:
            self.label_text.set(self.tinydict['Melanocytic nevi'])
        elif self.entry_text.get().find('Melanoma') != -1:
            self.label_text.set(self.tinydict['Melanoma'])
        elif self.entry_text.get().find('Benign keratosis-like lesions') != -1:
            self.label_text.set(self.tinydict['Benign keratosis-like lesions'])
        elif self.entry_text.get().find('Basal cell carcinoma') != -1:
            self.label_text.set(self.tinydict['Basal cell carcinoma'])
        elif self.entry_text.get().find('Actinic keratoses') != -1:
            self.label_text.set(self.tinydict['Actinic keratoses'])
        elif self.entry_text.get().find('Vascular lesions') != -1:
            self.label_text.set(self.tinydict['Vascular lesions'])
        elif self.entry_text.get().find('Dermatofibroma') != -1:
            self.label_text.set(self.tinydict['Dermatofibroma'])


    def do_identify(self):
        try:
            self.m = load_model("./model")
            print(self.e1.get())
            inputImgPath = self.e1.get()
            result = predict(self.m, inputImgPath)
            print(result)
            self.e2.delete(0, tk.END)
            self.e2.insert(0, 'result：'+result)
        except Exception as e:
            print(e)
            tkinter.messagebox.showwarning(title='failed', message='something has gone wrong')



if __name__ == '__main__':
    win = Frame()
    win.mainloop()
