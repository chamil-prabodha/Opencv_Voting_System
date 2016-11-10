import cv2
import numpy as np
from Tkinter import *
from tkinter import ttk
import Tkconstants
from PIL import Image
from PIL import ImageTk
import tkFileDialog
import os
import ImageTransform
import ImageSegmentaion
import Processing
import threading

mutex = threading.Semaphore(1)
arr = dict()
parties_images = []
path = ''
ballot_path = ''
increment = 0

root = Tk()
root.wm_title('Vote Counting Demo')
topframe = Frame(root,relief='ridge',width=100,height=100)
topframe.pack()

bottomframe = Frame(root)
bottomframe.pack(side='bottom')

frame = Frame(root)
frame.pack(side='bottom')
# root.minsize(width=400,height=500)
panelA = None
panelB = None

prg_bar = ttk.Progressbar(bottomframe,orient=HORIZONTAL,length = 200,mode='indeterminate')
prg_bar.grid(row=1,columnspan=2,padx=10,pady=10)
# ballot_paper = cv2.imread('001A.jpg')
# # P1 = cv2.imread('img/Party_1.jpg')
# img_main = ballot_paper.copy()

transform = ImageTransform.ImageTransform()
segmentation = ImageSegmentaion.ImageSegmentation()
processing = Processing.Processing(panelA,panelB,topframe,prg_bar)

# Select image for feature matching
def select_image():
    # Declare global variables
    global panelA,panelB,img,im_canny,topframe,path,ballot_path,prg_bar

    # prg_bar.start(50)
    if len(path)>0 and len(ballot_path)>0:
        count = len([file for file in os.listdir(ballot_path) if file.endswith('.jpg')])

        for file in os.listdir(ballot_path):
            if file.endswith('.jpg'):
                loc_path = ballot_path+'/'+file
                print('processing image: '+str(file))
                if len(loc_path)>0:
                    ballot_paper = cv2.imread(loc_path)
                    # P1 = cv2.imread('img/Party_1.jpg')
                    img_main = ballot_paper.copy()
                    # -------------------<Party Voting>--------------------------------------------------------------------------
                    img_main = segmentation.segmentPartyVoting(img_main)
                    img = img_main

                    # Display the cropped image
                    img_main = cv2.resize(img_main, (0, 0), fx=0.25, fy=0.25)
                    # cv2.imshow('Main', img_main)

                    # Display the edged image
                    # cannyout = cv2.resize(cannyout,(0,0),fx=0.25,fy=0.25)
                    # cv2.imshow('Canny',cannyout)

                    # ------------------</Party Voting>-------------------------------------------------------------------------
                    # -------------------<Preference Voting>--------------------------------------------------------------------
                    img_main = ballot_paper.copy()
                    pref_im = segmentation.segmentPreferenceVoting(img_main)

                    pref_gray = cv2.cvtColor(pref_im, cv2.COLOR_BGR2GRAY)

                    pref_gray_erode = cv2.erode(pref_gray.copy(), np.ones((3, 3), np.uint8), iterations=1)

                    pref_gray_erode = cv2.blur(pref_gray_erode, (3, 3), 0)

                    # pref_gray_erode = pref_gray
                    pref_cannyout = cv2.Canny(pref_gray_erode, 100, 200, 5)

                    pref_im2, pref_contours, pref_hierarchy = cv2.findContours(pref_cannyout.copy(), cv2.RETR_TREE,
                                                                               cv2.CHAIN_APPROX_SIMPLE)

                    # Select the contours with convexity and four points
                    pref_contours = [contour for contour in pref_contours if
                                     cv2.contourArea(contour) > 20 and cv2.contourArea(contour) < 10000 and len(
                                         cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)) == 4]

                    pref_contours = sorted(pref_contours, key=lambda x: (cv2.contourArea(x)), reverse=True)
                    # print(len(pref_contours))
                    # cv2.drawContours(pref_im, pref_contours, -1, (0, 0, 255), 3)

                    rect = []
                    centroids = []
                    for cnt in pref_contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(pref_im, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        rect.append([x, y, w, h])
                        cx = x + w / 2
                        cy = y + h / 2
                        centroids.append([cx, cy])

                    temp = []
                    for cx, cy in centroids:
                        if [cx, cy] not in temp:
                            temp.append([cx, cy])

                    centroids = temp
                    centroids = sorted(centroids, key=lambda x: (x[1], x[0]))

                    # print(len(centroids))
                    # print(centroids)

                    height, width = pref_im.shape[:2]
                    blocks = [None] * 40
                    for i in range(0, 4):
                        for j in range(0, 10):
                            for cx, cy in centroids:
                                index = i * 10 + j
                                if cx < (j + 1) * width / 10 and cx > j * width / 10 and cy < (
                                    i + 1) * height / 4 and cy > i * height / 4:
                                    blocks[i * 10 + j] = [cx, cy]
                                    # print(blocks[i])
                    # print(blocks)

                    voted = [i + 1 for i, val in enumerate(blocks) if val is None]
                    print(voted)
                    # cv2.imshow('im', pref_im)

                    # -------------------</Preference Voting>--------------------------------------------------------------------
                    # increment+=1
                    process_images(img,path,voted)

    # prg_bar.stop()
    # if panelA is not None:
    #     panelA.grid_forget()
    # if panelB is not None:
    #     panelB.grid_forget()

    # Open the file
    # path = tkFileDialog.askopenfilename()
    #
    # if len(path) >0:
    #     t1.delete(0,END)
    #     t1.insert(0,path)
    #
    #     template_ori = cv2.imread(path,0)
    #     img_0 = img.copy()
    #
    #     processing.startDetectPartyImage(img_0,template_ori)

# -------------------------------------------------------------------------------------------------------------

def clear_arr():
    for key,val in arr.iteritems():
        val[1] = 0
        for k,v in val[0].iteritems():
            val[0][k] = 0

def write_to_file():
    global arr
    prg_bar.stop()
    if not os.path.exists('output'):
        os.makedirs('output')

    with open('output/final_count.txt','w') as file:
        for key,val in arr.iteritems():
            file.write(str(key)+' - votes:'+str(val[1])+'\n')
            for k,v in val[0].iteritems():
                file.write('\t'+str(k)+': '+str(v)+'\n')

    print('file written')

def main_thread_start():
    clear_arr()
    prg_bar.start()
    # select_image()
    t = threading.Thread(target=select_image, name='select_image')
    t.start()

def select_party_symbols(t2):
    global panelA,panelB,img

    # if panelA is not None:
    #     panelA.grid_forget()
    # if panelB is not None:
    #     panelB.grid_forget()

    # Open directory
    p = selct_signs()
    t2.delete(0, END)
    t2.insert(0, p)
    # Copy the main image
    # img_0 = img.copy()


def process_images(img,path,voted):
    global mutex,arr,prg_bar
    img_0 = img.copy()
    processing.startDetectPartyVote(img, img_0, path, mutex, arr,voted)

def selct_signs():
    global path
    parties_images_ = []
    path = tkFileDialog.askdirectory()

    if len(path) > 0:
        for file in os.listdir(path):
            if file.endswith('.jpg'):
                filepath = path + '/' + file
                if len(filepath) > 0:
                    # template_ori = cv2.imread(filepath, 0)
                    arr[str(file)] = [dict(),0]
    print(path)
    print(arr)
    return path

def select_ballot_papers(t1):
    global ballot_path

    ballot_path = tkFileDialog.askdirectory()

    t1.delete(0, END)
    t1.insert(0, ballot_path)

    return ballot_path
# -----------------------------------------------------------------------------------------------------------------


l1 = Label(frame,text='Ballot Papers: ',relief='ridge',width=20).grid(row=0,column=0,padx=10,pady=10)
t1 = Entry(frame,width=100)
t1.grid(row=0,column=1,padx=10,pady=10)
btn1 = Button(frame,text='Browse',command = lambda :select_ballot_papers(t1),width=10)
# btn1.pack(side='right',fill=Tkconstants.NONE,expand='yes',padx='10',pady='10',)
btn1.grid(row=0,column=2,padx=10,pady=10)

l2 = Label(frame,text='Party Symbols: ',relief='ridge',width=20).grid(row=1,column=0,padx=10,pady=10)
t2 = Entry(frame,width=100)
t2.grid(row=1,column=1,padx=10,pady=10)
btn2 = Button(frame, text='Browse', command = lambda :select_party_symbols(t2), width=10)
# btn2.pack(side='right',fill=Tkconstants.NONE,expand='yes',padx='10',pady='10')
btn2.grid(row=1,column=2,padx=10,pady=10)

btn3 = Button(bottomframe, text='Count votes', command = main_thread_start, width=10)
# btn3.pack(side='right',fill='both',expand='yes',padx='10',pady='10')
btn3.grid(row=0,column=0,padx=10,pady=10)

btn4 = Button(bottomframe, text='Write to file', command = write_to_file, width=10)
# btn3.pack(side='right',fill='both',expand='yes',padx='10',pady='10')
btn4.grid(row=0,column=1,padx=10,pady=10)



root.mainloop()

# cv2.waitKey(0)
