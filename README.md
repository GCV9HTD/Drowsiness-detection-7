# Drowsiness-detection
Python Project driver drowsiness detection

In this project , I used OpenCV for gathering the images from webcam and feed them into a Deep Learning model which classifies whether the person's eyes are 'open' or 'closed' .

The approach used for this python Project is as follows:

Step1 - Take images as input from camera.

Step2 - Detect the face in the image and create a Region of Interest(ROI).

Step3 - Detect the eyes from ROI and feed it t the classifier.

Step4 - Classifier will categorize whether eyes are open or closed.

Step5 - Calculate score to check whether the person is drowsy.

# The Dataset

The dataset used for this model is taken from website of data-flair.training/python-project-driver-drowsiness
It comprises of 7000 images of people's eyes ubder different light conditions . After traning the model on the dataset , the final weights and model architectural file is attached as "models/cnnCat2.h5"

# The Model Architecture

The model used is  built with Keras using Convolutonal Neural networks(CNN). A convolutional neural network is a special type of deep neural network which performs extremely well for image classifictaion purposes.
A CNN basically consists of an input layer, an output layer and some hidden layers which can have multiple layers. A convolutional operation is performed on these layers using a filter that performs 2D matrix multiplication on the layer and filter.

#Prerequisites

The requirement for this is a webcam through which we will capture images . You need to have Python (version 3.6 )
installed on your system , then using pip, you can install the necesaary packages.

1. OpenCV - pip insatll opencv-python (face and eye detection).

2. TensorFlow - pip install tensorflow (keras uses TensorFlow as backend).

3. Keras - pip install keras (to build our classificaton model).

4. Pygame - pip install pygame (to play alarm sound).

# Contents

"haar cascade files" folder consists of the xml files that are needed to detect objects from the image. In our case, we are detecting the face and eyes of the person .

The models folder contains our model file "cnnCat2.h5" which was trained on convolutional neural metworks.

It also contains an audio clip "alarm.wav" which is played when the person is feeling drowsy.

"Model.py" file conatins the program through which we built our classification model by training our dataset. The implementation of convolutional neural netwrok lies in this file

"Drowsiness detection.py" is the main file of our project. To start the detection procedure , we have to run this file.

# Algorithm 

Step1 - Take images as Input from a Camera

With a webcam , we will take images as input. So to access the webcam, we made an infinite loop that will capture each frame. We use the method provided by OpenCV, cv2.VideoCapture(o) to access the camera and set the capture object(cap).
cap.read() will read each frame and we store the image in a frame variable.

Step2 - detect Face in the Image abd Create a region of Interest(ROI)

To detect the face in the image , we need to first convert the image into grayscale as the OpenCV algorithm for object detection takes gray images as input. We don't need color information to detect the objects. We will be using haar 
cascae classsifier to detect faces. This line is used to set our classifier face = cv2.CascadeClassifier ('path to our haar cascade xml file'). Then we perform the detection using faces = face.detectMultiScale(gray). It returnss an array of detections with x,y coordinates , and height , the width of the boundary box of the object.Now we can iterate over the faces and draw boundary boxes for each face.

Step3 - Detect the eyes from ROI and feed it to the classifier 

The same procedure to detect faces is used to detect eyes. first, we set the cascade classifier for eyes in leye and reye respectively then detect the eyes using left_eye=leye.detectMultiSclae(gray). Now we need to extract only the eyes data from the full image. This can be achieved by extracting the boundary box of the ye and then we can pull out the eye image from the frame with this code.
leye only contains the image data of the eye. This will be fed into our CNN classifier which will predict if eyes are open or closed . Similarly, we will be extracting the right eye into r_eye.

Step4 - Classifier will Categorize whether Eyes are Open or Closed
We are using CNN classifier for predicting the eye status. To feed our image into the model, we need to perform certain operations because the model needs the correct dimensions to start with. First, we convert the color image into grayscale using r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY). Then, we resize the image to 24*24 pixels as our model was trained on 24*24 pixel images cv2.resize(r_eye, (24,24)). We normalize our data for better convergence r_eye = r_eye/255 (All values will be between 0-1). Expand the dimensions to feed into our classifier. We loaded our model using model = load_model(‘models/cnnCat2.h5’) . Now we predict each eye with our model
lpred = model.predict_classes(l_eye). If the value of lpred[0] = 1, it states that eyes are open, if value of lpred[0] = 0 then, it states that eyes are closed.

'Step5' - Calculate Score to Check whether Person is Drowsy
The score is basically a value we will use to determine how long the person has closed his eyes. So if both eyes are closed, we will keep on increasing score and when eyes are open, we decrease the score. We are drawing the result on the screen using cv2.putText() function which will display real time status of the person.A threshold is defined for example if score becomes greater than 15 that means the person’s eyes are closed for a long period of time. This is when we beep the alarm using sound.play()

# How to run the program

Open the command prompt, go to the directory where our main file "drowsiness detection.py" exists .
Run the script with this command.

python "drowsiness detection.py"

It may takes a few seconds to open the webcam and start detection.

# Summary

In this Python project, we have built a drowsy driver alert system that you can implement in numerous ways. We used OpenCV to detect faces and eyes using a haar cascade classifier and then we used a CNN model to predict the status.



