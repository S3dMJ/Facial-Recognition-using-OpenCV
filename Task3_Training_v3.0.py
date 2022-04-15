import face_recognition as fr
import os
import pickle

datasetDirectory = "Images"

encodedFaces = []
labels = []
i = 0
for label in os.listdir(datasetDirectory):
    for filename in os.listdir(f"{datasetDirectory}/{label}"):
        image = fr.load_image_file(f"{datasetDirectory}/{label}/{filename}")
        print(f"{datasetDirectory}/{label}/{filename}")
        encoding = fr.face_encodings(image)
        encodedFaces.append(encoding)
        labels.append(label)
        print("\033[0;31m", len(encodedFaces[i]), "\033[0m")
        i = i + 1

print(len(encodedFaces))
data = {"encodedFaces": encodedFaces, "labels": labels}
with open('encodings.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
