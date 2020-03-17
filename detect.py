import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

if __name__ == "__main__":
	
	while True:
		_, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(gray)
		for face in faces:
			# x, y = face.left(), face.top()
			# x1, y1 = face.right(), face.bottom()
			# cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 1)

			landmarks = predictor(gray, face)
			left_mark = (landmarks.part(36).x, landmarks.part(36).y)
			right_mark = (landmarks.part(39).x, landmarks.part(39).y)
			top_mark = midpoint(landmarks.part(37), landmarks.part(38))
			bottom_mark = midpoint(landmarks.part(41), landmarks.part(40))

			horizontal = cv2.line(frame, left_mark, right_mark, (0, 255, 0), 1)
			vertical = cv2.line(frame, top_mark, bottom_mark, (0, 255, 0), 1)


		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1)
		if(key == 27):
			break

	cap.release()
	cv2.destroyAllWindows()
