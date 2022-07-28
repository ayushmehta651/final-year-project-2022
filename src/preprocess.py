import os
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import numpy as np
import cv2


def process():
    global fileName
    img = cv2.imread('/Users/ayushmehta/Desktop/htr_cnn_lstm/data/1.png',cv2.cv2.IMREAD_GRAYSCALE)
    pxmin = np.min(img)
    pxmax = np.max(img)
    imgContrast = (img - pxmin) / (pxmax - pxmin) * 255
    kernel = np.ones((3, 3), np.uint8)
    imgMorph = cv2.erode(imgContrast, kernel, iterations = 1)
    cv2.imwrite('/Users/ayushmehta/Desktop/htr_cnn_lstm/data/out.png', imgMorph)
    messagebox.showinfo("HW recognition", 'successfully written processed image')



def open_img():
    global fileName
    global load
    global img
    global render
    fileName = askopenfilename(initialdir='/Users/ayushmehta/Desktop/htr_cnn_lstm/data', title='Select image for analysis ',filetypes=[('image files', '.*')])
    #load = Image.open(fileName)
    print(fileName)
    messagebox.showinfo("HW recognition", 'successfully loaded image')

window = tk.Tk()

window.title("Handwritten Image Preprocessing")
window.geometry("500x510")
window.configure(background ="lightgreen")
title = tk.Label(text="Click below to choose HW image....", background = "lightgreen", fg="Brown", font=("", 15))
title.grid()
button1 = tk.Button(text="select HW image", command = open_img)
button1.grid(column=0, row=2, padx=10, pady = 10)
button2 = tk.Button(text="Preprocess", command=process)
button2.grid(column=0, row=3, padx=10, pady = 10)
window.mainloop()
