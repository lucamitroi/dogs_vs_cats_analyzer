import matplotlib.pyplot as plt
from PIL import ImageTk, Image
import PIL.Image
from keras.saving.save import load_model
from tkinter import *
from tkinter import filedialog
import cv2

loaded_model = load_model('dogs_vs_cats.h5')


def browse_files():
    filename = filedialog.askopenfilename(
        initialdir="/",
        title="Select a File",
        filetypes=(("Images", "*.*"),)
    )

    test_img = cv2.imread(filename)
    plt.imshow(test_img)
    test_img.shape
    test_img = cv2.resize(test_img, (128, 128))
    test_input = test_img.reshape((1, 128, 128, 3))

    image = PIL.Image.open(filename)
    image = image.resize((250, 300))
    test = ImageTk.PhotoImage(image)
    image_label = Label(image=test)
    image_label.image = test
    image_label.place(x=100, y=80)

    var_answer = StringVar()
    label_answer = Label(window, textvariable=var_answer)
    label_answer.config(font=("Courier", 13))
    if loaded_model.predict(test_input) == 1:
        var_answer.set("It's a dog")
    else:
        var_answer.set("It's a cat")
    label_answer.place(x=170, y=400)


window = Tk()
window.resizable(width=False, height=False)
window.title('Dog or Cat')
window.geometry("450x480")
window.config(background="#f1f1f0")

var = StringVar()
label = Label(window, textvariable=var)
label.config(font=("Courier", 13))
var.set("Browse a picture of your choosing:")
button_explore = Button(window, text="Browse Files", command=browse_files)

button_explore.place(x=350, y=20)
label.place(x=5, y=20)
window.mainloop()
