import cv2
import os
import numpy as np
from keras.models import model_from_json
import pygame
import random
pygame.mixer.init()


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# start the webcam feed
def emotion_detector():
    cap = cv2.VideoCapture(0)

    # pass here your video path
    # you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
    #cap = cv2.VideoCapture("D:\\Emotion_detection_with_CNN\\video\\1.mp4")
    flag=0


    # def on_key_press(event):
    #     print('Key {} was pressed'.format(event.name))
    #     if event.name=='space':
    #         #keyboard.unhook_all()
    #         flag=1
    #         sys.exit()
    # keyboard.wait()
    while flag==0:
        # Find haar cascade to draw bounding box around face
        grabbed, frame = cap.read()
        frame = cv2.resize(frame, (720, 540))
        if not grabbed:
            break
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        emotion_detected=0
        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            # keyboard.on_press(on_key_press) 
            # keyboard.wait()
            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            print(emotion_dict[maxindex])
            # keyboard.on_press(on_key_press) 
            emotion_detected=maxindex
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            if emotion_detected==0:
                dir="D:\Emotion_detection_with_CNN\songs\Angry"
                music_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".mp3")]
                selected_music = random.choice(music_files)
                pygame.mixer.music.load(selected_music)
                pygame.mixer.music.play()
                while cv2.waitKey(1) & 0xFF!= ord(' '):
                    continue

                # playsound("D:\ytubecode\Emotion_detection_with_CNN\angry1.mp3")
            elif emotion_detected==1:
                dir="D:\Emotion_detection_with_CNN\songs\Disgusted"
                music_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".mp3")]
                selected_music = random.choice(music_files)
                pygame.mixer.music.load(selected_music)
                pygame.mixer.music.play()
                while cv2.waitKey() & 0xFF!= ord(' '):
                    continue
                # playsound("D:\ytubecode\Emotion_detection_with_CNN\angry1.mp3")
            elif emotion_detected==2:
                dir="D:\Emotion_detection_with_CNN\songs\Fearful"
                music_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".mpeg")]
                selected_music = random.choice(music_files)
                pygame.mixer.music.load(selected_music)
                pygame.mixer.music.play()
                while cv2.waitKey(1) & 0xFF!= ord(' '):
                    continue
                # playsound("D:\ytubecode\Emotion_detection_with_CNN\angry1.mp3")
            elif emotion_detected==3:
                dir="D:\Emotion_detection_with_CNN\songs\Happy"
                music_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".mp3")]  
                selected_music = random.choice(music_files)
                pygame.mixer.music.load(selected_music)
                pygame.mixer.music.play()
                while cv2.waitKey(1) & 0xFF!= ord(' '):
                    continue
                # playsound("D:\ytubecode\Emotion_detection_with_CNN\angry1.mp3")
            elif emotion_detected==4:
                dir="D:\Emotion_detection_with_CNN\songs\\Neutral"
                music_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".mp3")]
                selected_music = random.choice(music_files)
                pygame.mixer.music.load(selected_music)
                pygame.mixer.music.play()
                while cv2.waitKey(1) & 0xFF!= ord(' '):
                    continue
                # playsound("D:\ytubecode\Emotion_detection_with_CNN\angry1.mp3")
            elif emotion_detected==5:
                dir="D:\Emotion_detection_with_CNN\songs\Sad"
                music_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".mp3")]
                selected_music = random.choice(music_files)
                pygame.mixer.music.load(selected_music)
                pygame.mixer.music.play()
                while cv2.waitKey(1) & 0xFF!= ord(' '):
                    continue
            elif emotion_detected==6:
                dir="D:\Emotion_detection_with_CNN\songs\Surprised"
                music_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".mp3")]
                selected_music = random.choice(music_files)
                pygame.mixer.music.load(selected_music)
                pygame.mixer.music.play()
                while cv2.waitKey(1) & 0xFF!= ord(' '):
                    continue
            
            break
    # keyboard.wait()
    cap.release()
    cv2.destroyAllWindows()
emotion_detector()
    



