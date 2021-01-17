import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
Tk().withdraw()

faceLocationXml = cv2.CascadeClassifier("./faceCascade.xml") # this is the face location xml file
cam = cv2.VideoCapture(0) # feed image with live Camera
# cam = cv2.VideoCapture("./video.mp4") # feed video from local file
cam.set(cv2.CAP_PROP_FPS,30)
cam.set(cv2.CAP_PROP_BUFFERSIZE,2)

# video write-------------
video_format = cv2.VideoWriter_fourcc(*'MJPG')
save = cv2.VideoWriter("video.mp4", video_format, 20.0, (640,480))
#---------

#------- for static image purpose-------
# img = askopenfilename()
# img = cv2.imread(img)
#--------------------------------------
while(cam.isOpened()):
	# reading images from camera
	ret, img = cam.read()
	filp_img = cv2.flip(img, 1)

	#--- if video file show rotated-----------
	# filp_img = cv2.resize(filp_img, (filp_img.shape[0]//4,filp_img.shape[1]//4))
	# filp_img = cv2.rotate(filp_img,2)
	#----------------------------------------
	############################
	gray = cv2.cvtColor(filp_img, cv2.COLOR_BGR2GRAY)
	faces = faceLocationXml.detectMultiScale(gray, 1.3,5)

	cv2.namedWindow("camera", 0)
	cv2.putText(filp_img, "Press 'Q' to Exit", (10,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
	# print(faces)

	for i,(x,y,w,h) in enumerate(faces):
		cv2.putText(filp_img, str(i+1), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
		# cv2.rectangle(filp_img, (x,y),(x+w, y+h), (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)),2)

	## swap two faces
	roi1 = 0
	roi2 = 0
	try:
		face1 = faces[0]
		face2 = faces[1]

		roi1 = filp_img[face1[1]: face1[1]+face1[3], face1[0]: face1[0]+face1[2]]
		roi1Copy = roi1.copy()
		roi2 = filp_img[face2[1]: face2[1]+face2[3], face2[0]: face2[0]+face2[2]]

		filp_img[face1[1]: face1[1]+face1[3], face1[0]: face1[0]+face1[2]] = cv2.resize(roi2[0:roi2.shape[0], 0:roi2.shape[1]],
			(filp_img[face1[1]: face1[1]+face1[3], face1[0]: face1[0]+face1[2]].shape[0],
				filp_img[face1[1]: face1[1]+face1[3], face1[0]: face1[0]+face1[2]].shape[1]))

		filp_img[face2[1]: face2[1]+face2[3], face2[0]: face2[0]+face2[2]] = cv2.resize(roi1Copy[0:roi1.shape[0], 0:roi1.shape[1]],
			(filp_img[face2[1]: face2[1]+face2[3], face2[0]: face2[0]+face2[2]].shape[0],
				filp_img[face2[1]: face2[1]+face2[3], face2[0]: face2[0]+face2[2]].shape[1]))

		# filp_img[face2[1]: face2[1]+face2[3], face2[0]: face2[0]+face2[2]] = [np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)]
		


	except Exception:
		pass
	####################

	# save.write(filp_img) # for saving video
	cv2.imshow("camera", filp_img)
	# cv2.waitKey()
	if cv2.waitKey(1) & 0xff == ord('q'):
		break