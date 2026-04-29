import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator 


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


FRAME_SKIP = 3  
REFRESH_MS = 33 

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu')) 
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.load_weights(r'C:\Users\farah\Desktop\cvecu\face detection to emoji project\src\model.weights.h5')
cv2.ocl.setUseOpenCL(False)

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprised"
}

# Base path
cur_path = os.path.dirname(os.path.abspath(__file__))

# Emoji images
emoji_dist = {
    0: os.path.join(cur_path, "emojis/angry.png"),
    1: os.path.join(cur_path, "emojis/disgusted.png"),
    2: os.path.join(cur_path, "emojis/fearful.png"), 
    3: os.path.join(cur_path, "emojis/happy.png"),
    4: os.path.join(cur_path, "emojis/neutral.png"),
    5: os.path.join(cur_path, "emojis/sad22.png"),
    6: os.path.join(cur_path, "emojis/surprised.png")
}

# Initialize variables globally before mainloop starts
global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
cap1 = None 
global show_text
show_text = [0]
global frame_counter
frame_counter = 0 
global last_emotion_index
last_emotion_index = show_text[0] 

root = None
lmain = None
lmain2 = None
lmain3 = None

def show_subject():
    # Variables are correctly declared as global here, as they are modified/used
    global cap1, frame_counter, last_frame1, show_text, root, lmain, last_emotion_index, REFRESH_MS

    if cap1 is None:
        cap1 = cv2.VideoCapture(0)
        if not cap1.isOpened():
            print("Can't open the camera")
            if root:
                root.after(REFRESH_MS, show_subject)
            return

    flag1, frame1 = cap1.read()
    if not flag1:
        if root:
            lmain.after(REFRESH_MS, show_subject)
        return

    frame1 = cv2.resize(frame1, (600, 500))

    if not hasattr(show_subject, 'bounding_box'):
        show_subject.bounding_box = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    bounding_box = show_subject.bounding_box

    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5) 
    
    current_emotion = last_emotion_index 
    
    # --- FRAME SKIPPING LOGIC ---
    if frame_counter % FRAME_SKIP == 0:
        
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

            roi_gray_frame = gray_frame[y: y + h, x: x + w]
            if roi_gray_frame.size == 0:
                continue
                
            cropped_img = np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1)
            cropped_img = cropped_img / 255.0 
            
            prediction = emotion_model.predict(np.expand_dims(cropped_img, axis=0), verbose=0)
            current_emotion = int(np.argmax(prediction[0]))
            
            show_text[0] = current_emotion 
            last_emotion_index = current_emotion
            
            cv2.putText(frame1, emotion_dict[current_emotion], (x+20, y-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            cv2.putText(frame1, emotion_dict[last_emotion_index], (x+20, y-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    frame_counter += 1
        
    # Update video frame display
    last_frame1 = frame1.copy()
    pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    
    lmain.after(REFRESH_MS, show_subject)


def show_avatar():
    # Variables are correctly declared as global here
    global root, lmain2, lmain3, emoji_dist, show_text, REFRESH_MS

    emotion_key = show_text[0]
    frame2 = cv2.imread(emoji_dist[emotion_key])
    
    if frame2 is None:
        print(f"Error: Emoji image not found at {emoji_dist[emotion_key]}")
        if root:
            lmain2.after(REFRESH_MS, show_avatar)
        return

    frame2 = cv2.resize(frame2, (400, 400)) 

    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(pic2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    
    lmain2.imgtk2 = imgtk2
    lmain2.configure(image=imgtk2)
    lmain3.configure(text=emotion_dict[emotion_key])

    lmain2.after(REFRESH_MS, show_avatar)


if __name__=='__main__':
    root=tk.Tk() 
    
    # --- FIX APPLIED: REMOVED GLOBAL DECLARATION HERE ---

    lmain = tk.Label(master=root,padx=50,bd=10)
    lmain2 = tk.Label(master=root,bd=10)
    lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black', font=('arial', 45, 'bold'))

    lmain.pack(side=LEFT)
    lmain.place(x=50,y=250)
    lmain3.pack()
    lmain3.place(x=960,y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900,y=350)

    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")
    root['bg']='black'
    
    exitButton = Button(root, text='Quit',fg="red", command=root.destroy, font=('arial',25,'bold')).pack(side = BOTTOM)
    
    show_subject()
    show_avatar()
    
    root.mainloop()