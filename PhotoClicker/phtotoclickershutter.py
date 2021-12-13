import cv2 as cv
import numpy as np

# getting video path
video = cv.VideoCapture(0)
a = 0

eye = cv.CascadeClassifier('cascadeClassifier/haarcascade_eye.xml')
# checking correct video path

if(video.isOpened() == False):
    print("video path not correct")

else:
    while(video.isOpened()):
        flag, frame = video.read()

        if(flag == True):
            gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            
            cropped_eye =frame
            eyes = eye.detectMultiScale(gray)
            f=0
            for (ex, ey, ew, eh) in eyes:
                # cv.rectangle(gray, (ex,ey), (ex+ew, ey+eh), (0, 0, 255), 2)
                cropped_eye = gray[ey:ey+eh, ex:ex+ew]
                f+=1
                # print(f)
                cv.imshow('Saving', gray)

            if(f==2): #increasing accuracy of eye occurence
                a+=1
                a.__str__
                print("captured "+ str(a))
                path ="cvSavedPhotos/"
                cropped_eye=cv.resize(cropped_eye, (100, 100))
                cv.imwrite(str(path) + str(a) +".png", cropped_eye)
                
            k = cv.waitKey(1) & 0xFF
            # press esc to exit
            if k == 27:
                break
        else:
            break
cv.destroyAllWindows