import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import mediapipe as mp
import math

padding = 20

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Create Tkinter window
window = tk.Tk()
window.title("Mask Detection")
window.geometry("400x450")

# Create a label to display the selected image
img_label = tk.Label(window)
img_label.pack(pady=10)


# Add Gaze Detection
def eye_gaze(image):
    mp_face_mesh = mp.solutions.face_mesh
    # BOTH eyes indices
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7,
                               min_tracking_confidence=0.5) as face_mesh:
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        mesh_points = np.array(
            [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

        TLeye = mesh_points[33]
        TReye = mesh_points[133]

        TRLeye = mesh_points[362]
        TRReye = mesh_points[263]

        TDeye = mesh_points[145]
        TUeye = mesh_points[159]


        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
        center_right = np.array([l_cx, l_cy], dtype=np.int32)
        center_left = np.array([r_cx, r_cy], dtype=np.int32)


        print(math.dist(TRLeye, center_right))

        if math.dist(TRLeye, center_right) > 0.65 * (math.dist(TRReye, TRLeye)):
            return("right")

        elif math.dist(TReye, center_left) > 0.6 * math.dist(TReye, TLeye):
            return("left")

        elif math.dist(TDeye, center_left) > 0.55 * math.dist(TUeye, TDeye):
            return("down/up")

        else:
            return("null")
        


def highlightFace(net, image, conf_threshold=0.7):
    imageOpencvDnn=image.copy()
    imageHeight=imageOpencvDnn.shape[0]
    imageWidth=imageOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(imageOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*imageWidth)
            y1=int(detections[0,0,i,4]*imageHeight)
            x2=int(detections[0,0,i,5]*imageWidth)
            y2=int(detections[0,0,i,6]*imageHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(imageOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(imageHeight/150)), 8)
    return imageOpencvDnn,faceBoxes



# Create a button to select an image
def process_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        resized_image = cv2.resize(image, (150, 150))
        preprocessed_image = resized_image / 255.0
        input_image = np.expand_dims(preprocessed_image, axis=0)

        # Display the selected image
        img = Image.open(file_path)
        img = img.resize((250, 250), resample=Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        img_label.configure(image=img)  # Update the image in the label
        img_label.image = img

        prediction = model.predict(input_image)
        predicted_label = "with mask" if prediction[0] > 0.5 else "without mask"
        gaze_dir = eye_gaze(image)
        if predicted_label == "with mask":
            result_label.config(text=f"{predicted_label}\n")
        elif predicted_label == "without mask":
            resultImg, faceBoxes = highlightFace(faceNet, image)
            if not faceBoxes:
                result_label.config(text="No face detected")
            else:
                for faceBox in faceBoxes:
                    face = image[max(0, faceBox[1] - padding):
                                 min(faceBox[3] + padding, image.shape[0] - 1), max(0, faceBox[0] - padding)
                                 :min(faceBox[2] + padding, image.shape[1] - 1)]

                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]

                    ageNet.setInput(blob)
                    agePreds = ageNet.forward()
                    age = ageList[agePreds[0].argmax()]

                    result_label.config(text=f"Without mask\nGender: {gender}\nAge: {age[1:-1]} years\n")


select_button = tk.Button(window, text="Select Image", command=process_image)
select_button.pack(pady=10)

# Create a label for model selection
model_label = tk.Label(window, text="Created By Mahdi Savoji", font=("Arial", 8), bg='white')
model_label.pack(side="bottom")

# Create a label to display the result
result_label = tk.Label(window, text="")
result_label.pack(pady=10)

# Load the model and other variables
model = tf.keras.models.load_model('./my_InceptionV3.h5')
padding = 20

window.mainloop()