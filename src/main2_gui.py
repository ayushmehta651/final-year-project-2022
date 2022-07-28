from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
#import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
from WordSegmentation import wordSegmentation, prepareImg
import os
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
fileName=''
def open_img():
    global fileName
    global load
    global img
    global render
    fileName = askopenfilename(initialdir='F:\SimpleHTR-master\data', title='Select image for analysis ',filetypes=[('image files', '.png')])
    #load = Image.open(fileName)
    print(fileName)
    FilePaths.fnInfer=fileName
    messagebox.showinfo("HW recognition", 'successfully loaded image')
    print(FilePaths.fnInfer)
    #render = ImageTk.PhotoImage(load)
    #img = tk.Label(image=render, height="250", width="500")
    #img.image = render
    #img.place(x=0, y=0)
    #img.grid(column=0, row=1, padx=10, pady = 10)



class FilePaths:
        "filenames and paths to data"
        fnCharList = '../model/charList.txt'
        fnAccuracy = '../model/accuracy.txt'
        fnTrain = '../data/'
        fnInfer = fileName
        fnCorpus = '../data/corpus.txt'
decoderType = DecoderType.BestPath
print(open(FilePaths.fnAccuracy).read())
model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)
def train(model, loader):
	"train NN"
	epoch = 0 # number of training epochs since start
	bestCharErrorRate = float('inf') # best valdiation character error rate
	noImprovementSince = 0 # number of epochs no improvement of character error rate occured
	earlyStopping = 5 # stop training after this number of epochs without improvement
	while True:
		epoch += 1
		print('Epoch:', epoch)

		# train
		print('Train NN')
		loader.trainSet()
		while loader.hasNext():
			iterInfo = loader.getIteratorInfo()
			batch = loader.getNext()
			loss = model.trainBatch(batch)
			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

		# validate
		charErrorRate = validate(model, loader)
		
		# if best validation accuracy so far, save model parameters
		if charErrorRate < bestCharErrorRate:
			print('Character error rate improved, save model')
			bestCharErrorRate = charErrorRate
			noImprovementSince = 0
			model.save()
			open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
		else:
			print('Character error rate not improved')
			noImprovementSince += 1

		# stop training if no more improvement in the last x epochs
		if noImprovementSince >= earlyStopping:
			print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
			break


def validate(model, loader):
	"validate NN"
	print('Validate NN')
	loader.validationSet()
	numCharErr = 0
	numCharTotal = 0
	numWordOK = 0
	numWordTotal = 0
	while loader.hasNext():
		iterInfo = loader.getIteratorInfo()
		print('Batch:', iterInfo[0],'/', iterInfo[1])
		batch = loader.getNext()
		(recognized, _) = model.inferBatch(batch)
		
		print('Ground truth -> Recognized')	
		for i in range(len(recognized)):
			numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
			numWordTotal += 1
			dist = editdistance.eval(recognized[i], batch.gtTexts[i])
			numCharErr += dist
			numCharTotal += len(batch.gtTexts[i])
			print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
	
	# print validation result
	charErrorRate = numCharErr / numCharTotal
	wordAccuracy = numWordOK / numWordTotal
	print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
	return charErrorRate

text1 = []
def infer(model, fnImg):
	img = prepareImg(cv2.imread(fnImg), 50)
	#cv2.imshow('kdkd',img),cv2.waitKey(0)
	res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
	print('Recognized text as follows')
	for (j, w) in enumerate(res):
			(wordBox, wordImg) = w
			cv2.imwrite('../out/%d.png' %j, wordImg) # save word
			fn='../out/%d.png'%j
			#cv2.imshow('kkk',wordImg),cv2.waitKey(0)
			i=cv2.imread(fn,cv2.IMREAD_GRAYSCALE)
			img = preprocess(i, Model.imgSize)
			#cv2.imshow('kkk',i),cv2.waitKey(0)
			batch = Batch(None, [img])
			(recognized, probability) = model.inferBatch(batch, True)
			print(recognized[0],end=' ')
			print('Probability:', probability[0])
			text1.append(recognized[0])
	print("\n")
	print(' '.join(text1))
	with open('result.txt','a') as f:
	    for line in text1:
	        f.write(' '+line)



def main():
    infer(model, FilePaths.fnInfer)
window = tk.Tk()

window.title("Handwritten Recognition using CNN")
window.geometry("500x510")
window.configure(background ="lightgreen")
title = tk.Label(text="Click below to choose HW image....", background = "lightgreen", fg="Brown", font=("", 15))
title.grid()
button1 = tk.Button(text="select HW image", command = open_img)
button1.grid(column=0, row=2, padx=10, pady = 10)
button2 = tk.Button(text="Recognize", command=main)
button2.grid(column=0, row=3, padx=10, pady = 10)
window.mainloop()

#if __name__ == '__main__':
#	main()

