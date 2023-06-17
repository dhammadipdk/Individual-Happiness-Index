# TECHBARK
from keras.backend import set_session
import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2
import json
from tkinter import messagebox as mb
from tkinter import *
from select import select
ls = []


class Quiz:
    def __init__(self):

        self.q_no = 0

        self.display_title()
        self.display_question()

        self.opt_selected = IntVar()

        self.opts = self.radio_buttons()
        self.display_options()

        self.buttons()

        self.data_size = len(question)

        self.correct = 0

    def display_result(self):

        wrong_count = self.data_size - self.correct
        correct = f"Done: 30"
        wrong = f"Outoff: 30"

        score = int(self.correct / self.data_size * 100)
        result = f"complete: 100%"

        mb.showinfo("See Terminal for result", f"{result}\n{correct}\n{wrong}\n")

    def check_ans(self, q_no):
        
        if self.opt_selected.get() == answer[q_no]:
            return True

        select_opt = self.opt_selected.get()
        ls.append(select_opt) 

    def next_btn(self):

        if self.check_ans(self.q_no):

            self.correct += 1

        self.q_no += 1

        if self.q_no == self.data_size:

            self.display_result()

            gui.destroy()
        else:
            self.display_question()
            self.display_options()

    def buttons(self):

        next_button = Button(gui, text="Next", command=self.next_btn,
                             width=10, bg="blue", fg="white", font=("ariel", 16, "bold"))

        next_button.place(x=350, y=380)

        quit_button = Button(gui, text="Quit", command=gui.destroy,
                             width=5, bg="black", fg="white", font=("ariel", 16, " bold"))

        quit_button.place(x=700, y=50)

    def display_options(self):
        val = 0

        self.opt_selected.set(0)

        for option in options[self.q_no]:
            self.opts[val]['text'] = option
            val += 1

    def display_question(self):

        q_no = Label(gui, text=question[self.q_no], width=60,
                     font=('ariel', 16, 'bold'), anchor='w')

        q_no.place(x=70, y=100)

    def display_title(self):

        title = Label(gui, text="Survey",
                      width=50, bg="green", fg="white", font=("ariel", 20, "bold"))

        title.place(x=0, y=2)

    def radio_buttons(self):

        q_list = []

        y_pos = 150

        while len(q_list) < 10:

            radio_btn = Radiobutton(gui, text=" ", variable=self.opt_selected,
                                    value=len(q_list)+1, font=("ariel", 14))

            q_list.append(radio_btn)

            radio_btn.place(x=100, y=y_pos)

            y_pos += 40

        return q_list


gui = Tk()

gui.geometry("800x450")

gui.title("Survey")

with open('data.json') as f:
    data = json.load(f)

question = (data['question'])
options = (data['options'])
answer = (data['answer'])

quiz = Quiz()

# DK

''' Keras took all GPU memory so to limit GPU usage, I have add those lines'''


config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.compat.v1.Session(config=config))
''' Keras took all GPU memory so to limit GPU usage, I have add those lines'''

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

video_capture = cv2.VideoCapture(0)
model = load_model('keras_model/model_5-49-0.62.hdf5')
model.get_config()

target = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 5)
        face_crop = frame[y:y + h, x:x + w]
        face_crop = cv2.resize(face_crop, (48, 48))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_crop = face_crop.astype('float32') / 255
        face_crop = np.asarray(face_crop)
        face_crop = face_crop.reshape(
            1, 1, face_crop.shape[0], face_crop.shape[1])
        result = target[np.argmax(model.predict(face_crop))]
        cv2.putText(frame, result, (x, y), font,
                    1, (200, 0, 0), 3, cv2.LINE_AA)
        print(result)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


gui.mainloop()

print(ls)
print(result)

import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

df=pd.read_csv(r'C:\Users\Acer\Desktop\HACKATHON\Train.csv')
df2=pd.read_csv(r'C:\Users\Acer\Desktop\HACKATHON\Test.csv')

var="Q46"

df.fillna(0, inplace=True)
df2.fillna(0,inplace=True)

df3=df[['Q1','Q2','Q5','Q6','Q27','Q46','Q47','Q48','Q49','Q50','Q51','Q54','Q90','Q106','Q108','Q119','Q120','Q121','Q131','Q142','Q143','Q158','Q160','Q164','Q173','Q176','Q199','Q240','Q253','Q260','Q273']]
df4=df2[['Q1','Q2','Q5','Q6','Q27','Q46','Q47','Q48','Q49','Q50','Q51','Q54','Q90','Q106','Q108','Q119','Q120','Q121','Q131','Q142','Q143','Q158','Q160','Q164','Q173','Q176','Q199','Q240','Q253','Q260','Q273']]

train_labels = pd.DataFrame(df3[var])
train_labels = np.array(df3[var])
train_features= df3.drop(var, axis = 1)
feature_list = list(train_features.columns)
train_features = np.array(train_features)

test_labels = pd.DataFrame(df4[var])
test_labels = np.array(df4[var])
# test_features= df4.drop(var, axis = 1)
test_features = np.array(ls)
test_features = test_features.reshape(1,30)

rf=RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1,random_state =42,max_features = "auto", min_samples_leaf = 12)
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)

print(predictions)
if(predictions==1.): print("Happiness Index: 9.3\n Congratulations! YOU Being a happy personality we hope you contiue this happiness.")
elif(predictions==2.): print("Happiness Index: 7.3\n Make a mind. Stress Little. Life is Fun!")
elif(predictions==3.): print("Happiness Index: 5.1\n Secure Yourself, Be with you dear ones. DO Exercise and dont be stressfull.")
elif(predictions==4.): print("Happiness Index: 2.3\n WARNING! You need to take holiday. Go for trip with family spend more time with them and do things which make you happy.")
else: print("Can't Predict please try again")