# Import required Libraries
from tkinter import *
from tkinter import messagebox as mb
import json
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
from keras.backend import set_session


# Create an instance of TKinter Window or frame
win= Tk()

# Set the size of the window
win.geometry("700x350")# Create a Label to capture the Video frames
label =Label(win)
label.grid(row=0, column=0)
# cap= cv2.VideoCapture(0)

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.compat.v1.Session(config=config))
''' Keras took all GPU memory so to limit GPU usage, I have add those lines'''

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

video_capture = cv2.VideoCapture(0)
model = load_model('keras_model/model_5-49-0.62.hdf5')
model.get_config()


# Define function to show frame
# def show_frames():
#     # Get the latest frame and convert into Image
#     cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(cv2image)

#     # Convert image to PhotoImage
#     imgtk = ImageTk.PhotoImage(image = img)
#     label.imgtk = imgtk
#     label.configure(image=imgtk)


target = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 5)
        face_crop = frame[y:y + h, x:x + w]
        face_crop = cv2.resize(face_crop, (48, 48))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_crop = face_crop.astype('float32') / 255
        face_crop = np.asarray(face_crop)
        face_crop = face_crop.reshape(1, 1, face_crop.shape[0], face_crop.shape[1])
        result = target[np.argmax(model.predict(face_crop))]
        cv2.putText(frame, result, (x, y), font, 1, (200, 0, 0), 3, cv2.LINE_AA)

    # Display the resulting frame
    # cv2.imshow('Video', frame)
    # Repeat after an interval to capture continiously
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()




#new screen
class Quiz:
    def __init__(self):
        self.q_no=0
        self.display_title()
		self.display_question()
		self.opt_selected=IntVar()
		self.opts=self.radio_buttons()
		self.display_options()
		self.buttons()

		self.data_size=len(question)        

	def buttons(self):
		
		# The first button is the Next button to move to the
		# next Question
		next_button = Button(gui, text="Next",command=self.next_btn,
		width=10,bg="blue",fg="white",font=("ariel",16,"bold"))
		
		# placing the button on the screen
		next_button.place(x=350,y=380)
		
		# This is the second button which is used to Quit the GUI
		quit_button = Button(gui, text="Quit", command=gui.destroy,
		width=5,bg="black", fg="white",font=("ariel",16," bold"))
		
		# placing the Quit button on the screen
		quit_button.place(x=700,y=50)


	# This method deselect the radio button on the screen
	# Then it is used to display the options available for the current
	# question which we obtain through the question number and Updates
	# each of the options for the current question of the radio button.
	def display_options(self):
		val=0
		
		# deselecting the options
		self.opt_selected.set(0)
		
		# looping over the options to be displayed for the
		# text of the radio buttons.
		for option in options[self.q_no]:
			self.opts[val]['text']=option
			val+=1


	# This method shows the current Question on the screen
	def display_question(self):q_no = Label(gui, text=question[self.q_no], width=60,
		font=( 'ariel' ,16, 'bold' ), anchor= 'w' )
		
		#placing the option on the screen
		q_no.place(x=70, y=100)


	# This method is used to Display Title
	def display_title(self):
		
		# The title to be shown
		title = Label(gui, text="GeeksforGeeks QUIZ",
		width=50, bg="green",fg="white", font=("ariel", 20, "bold"))
		
		# place of the title
		title.place(x=0, y=2)


	# This method shows the radio buttons to select the Question
	# on the screen at the specified position. It also returns a
	# list of radio button which are later used to add the options to
	# them.
	def radio_buttons(self):
		
		# initialize the list with an empty list of options
		q_list = []
		
		# position of the first option
		y_pos = 150
		
		# adding the options to the list
		while len(q_list) < 4:
			
			# setting the radio button properties
			radio_btn = Radiobutton(gui,text=" ",variable=self.opt_selected,
			value = len(q_list)+1,font = ("ariel",14))
			
			# adding the button to the list
			q_list.append(radio_btn)
			
			# placing the button
			radio_btn.place(x = 100, y = y_pos)
			
			# incrementing the y-axis position by 40
			y_pos += 40
		
		# return the radio buttons
		return q_list

# Create a GUI Window
gui = Tk()

# set the size of the GUI Window
gui.geometry("800x450")

# set the title of the Window
gui.title("GeeksforGeeks Quiz")

# get the data from the json file
with open("data.json", "r") as read_file:
   data = json.load(read_file)

# set the question, options, and answer
question = (data['question'])
options = (data['options'])
answer = (data[ 'answer'])

quiz = Quiz()
frame()
win.mainloop()
gui.mainloop()