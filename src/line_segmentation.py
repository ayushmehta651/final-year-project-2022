import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib as mpl
import traceback

mpl.rcParams['legend.fontsize'] = 10

pd.set_option('display.expand_frame_repr', False)
fn=0
path='./result/'

#Taking any image from the sample images
#In case of slanted image, straighten it using image-straighten.py, then use it
img = cv.imread('./data/21.jpeg')

# In[lineSegment]
#*****************************************************************************#
def lineSegment(img):
    #Binarization
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    th, threshed = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
   
    upper=[]
    lower=[]
    flag=True
    for i in range(threshed.shape[0]):
        #sub-matrix of size 1 * no_of_col
        col = threshed[i:i+1,:]
        cnt=0
        if flag:
            #checking last column pixel value
            cnt=np.count_nonzero(col)
            if cnt >0:
                upper.append(i)
                flag=False
                print(col.shape[0], col.shape[1])
        else:
            cnt=np.count_nonzero(col)
            if cnt <2:
                lower.append(i)
                flag=True
                print(col.shape[0], col.shape[1])

    textLines=[]
    if len(upper)!= len(lower):lower.append(threshed.shape[0])
    # print(upper)
    # print(lower)
    for i in range(len(upper)):
        timg=img[upper[i]:lower[i],:]
        
        if timg.shape[0]>5:
#            plt.imshow(timg)
#            plt.show()
            timg=cv.resize(timg,((timg.shape[1]*5,timg.shape[0]*8)))
            textLines.append(timg)

    return textLines

# main
try:
    textLines=lineSegment(img)
    print ('***** No. of Lines ***** : ',len(textLines))     

    counter = 22
    for lines in textLines:
        cv.imwrite("./data/" + str(counter) +".jpeg",lines)
        counter=counter+1
       
except Exception as e:
    print ('Error Message ',e)
    cv.destroyAllWindows()
    traceback.print_exc()
    pass

traceback.print_exc() 

