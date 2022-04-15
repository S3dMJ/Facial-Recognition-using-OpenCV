# EE490 Assignment Four

# Step One
In this task we are required to make a dataset of 6 members of the Jurassic Park cast and adding pictures of the team members. Downloading the 270 images manually will take a lot of time and effort. A programming library called Bing Image downloader was used in this step. This library enabled us to download all images in less than 15 minutes. The dataset was stored in a folder called “Images”. This folder has a nine folder, each folder is named after a person and has 30 images of that person.

![alt text](https://github.com/S3dMJ/EE490_A4/blob/main/Picture1.png?raw=true)

# Step Two
In the next step we are required to find the 128d feature vector before the training. This was done by putting all images in a nested for loop and finding each image feature vector by using the pre trained face_recognition library. All encodings and names were then appended to lists and then a dictionary. The dictionary was turned into a pickle file to be used in step three.

# Step Three
In this the step, the real time gathered from the webcam using cv2 library was then compared to the list of encodings to find a match. If a match was found the name of the match will be visible under the face region, and if no match was found then the phrase “unknown” will be visible. The following is an example of this step.

![alt text](https://github.com/S3dMJ/EE490_A4/blob/main/EE490_Task3_screenshot_14.04.2022.png?raw=true)
