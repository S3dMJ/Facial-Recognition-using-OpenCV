import face_recognition as fr
import pickle
import cv2
import imutils
import time
from imutils.video import VideoStream

with open('encodings.pickle', 'rb') as handle:
    data = pickle.load(handle)

encodedFaces = data["encodedFaces"]
faces = []
i = 0
while i < len(encodedFaces):
    faces.append(encodedFaces[i][0])
    i = i + 1

cap = VideoStream(src=0).start()
time.sleep(2.0)
while True:
    frame = cap.read()
    coloredFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    coloredFrame = imutils.resize(frame, width=750)
    rescaleFactor = frame.shape[1] / float(coloredFrame.shape[1])
    regionOfInterest = fr.face_locations(coloredFrame, model='hog')
    encodings = fr.face_encodings(coloredFrame, regionOfInterest)
    names = []
    for encoding in encodings:
        matches = fr.compare_faces(faces, encoding)
        name = "Unknown"
        if True in matches:
            matchedIds = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIds:
                name = data["labels"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
    for ((top, right, bottom, left), name) in zip(regionOfInterest, names):
        top = int(top * rescaleFactor)
        right = int(right * rescaleFactor)
        bottom = int(bottom * rescaleFactor)
        left = int(left * rescaleFactor)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
        cv2.putText(frame, name, (left+5, bottom+25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow('EE490_Task3', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
