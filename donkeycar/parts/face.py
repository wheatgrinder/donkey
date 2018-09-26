import cv2
import sys
import time
import numpy as np
from PIL import Image
class detect():
        def __init__(self, basedir, NETWORK_IMAGE_WIDTH = 224, NETWORK_IMAGE_HEIGHT = 224, draw_on_img=True, probability_threshold = 0.2, debug=False):
                



                
                self.NETWORK_IMAGE_WIDTH = NETWORK_IMAGE_WIDTH
                self.NETWORK_IMAGE_HEIGHT = NETWORK_IMAGE_HEIGHT
                self.BASE_DIR = basedir+'/'
                self.probability_threshold = probability_threshold
                self.on = True
                self.draw_on_img = draw_on_img
                self.debug = debug
                

                time.sleep(2)
                self.out_data='none'
                 # we need an image to start or 
                self.img_arr = np.zeros((1,1,3), np.uint8)
                self.out_img_arr = np.zeros((1,1,3), np.uint8)
                self.hldCnt = 0;   


                #self.cascPath = self.BASE_DIR + "haarcascade_frontalface_default.xml"
                #self.cascPath = self.BASE_DIR + "haarcascade_upperbody.xml"
                self.cascPath = self.BASE_DIR + "lbpcascade_frontalface.xml"
                
                self.faceCascade = cv2.CascadeClassifier(self.cascPath)   
                

                        
        def update(self):
                        
                while self.on:
                        gray = cv2.cvtColor(self.img_arr,cv2.COLOR_BGR2GRAY)
                        faces = self.faceCascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=(3, 3)
                        )

                        #if(len(faces)!=0):
                        #    print("faces: "+str(len(faces)))
                        
                        for (x, y, w, h) in faces:
                            cv2.rectangle(self.img_arr, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        self.out_img_arr = self.img_arr

        
        def run_threaded(self,img_arr):
                self.img_arr = img_arr
                return self.out_img_arr, self.out_data
                        


        def shutdown(self):
                # indicate that the thread should be stopped
                self.on = False
                print('stoping face')
                
                #Clean up
                time.sleep(.5)
                #self.stream.close()
                #self.rawCapture.close()
                #self.camera.close()


