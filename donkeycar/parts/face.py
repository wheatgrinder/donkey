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
                self.out_data=[0,0]
                self.center_of_img = [0,0]
                 # we need an image to start or 
                self.img_arr = np.zeros((1,1,3), np.uint8)
                self.out_img_arr = np.zeros((1,1,3), np.uint8)
                self.hldCnt = 0; 



                #self.cascPath = self.BASE_DIR + "haarcascade_frontalface_default.xml"
                #self.cascPath = self.BASE_DIR + "haarcascade_upperbody.xml"
                self.cascPath = self.BASE_DIR + "lbpcascade_frontalface.xml"
                
                self.faceCascade = cv2.CascadeClassifier(self.cascPath)   
                self.faces_found =[]

        def update(self):
                        
                while self.on:
                        img_h, img_w = self.img_arr.shape[:2]
                        self.center_of_img = (int(img_w/2), int(img_h/2))
                        
                        gray = cv2.cvtColor(self.img_arr,cv2.COLOR_BGR2GRAY)
                        faces = self.faceCascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=(3, 3)
                        )

                        #if(len(faces)!=0):
                        #    print("faces: "+str(len(faces)))
                        #lets faces detected stay until new faces enter.. 
                        #print("len of faces:", len(faces))
                        if len(faces)>0:
                                self.faces_found = faces.copy()


                        for (x, y, w, h) in self.faces_found:
                            center=(int((x+(x+w))/2), int(((y+h)+y)/2))
                            self.out_data = (center)
                            if self.draw_on_img: 
                                cv2.rectangle(self.img_arr, (x, y), (x+w, y+h), (255, 0, 0), 2)
                                #(x,y) = (x2 + x1)/2, (y2+y1)/2
                                cv2.circle(self.img_arr, center, 20, (255,0,0), 2)  
                                cv2.circle(self.img_arr, self.center_of_img, 30, (0,0,255), 2) 
                               
                        
                        
                       

                        self.out_img_arr = self.img_arr
                        if len(self.out_data) == 2:
                                self.face_x = self.out_data[0]
                                self.face_y = self.out_data[1]
                        if len(self.center_of_img) ==2:
                                self.img_x = self.center_of_img[0]
                                self.img_y = self.center_of_img[1]
                        
                        if self.debug:
                                print(self.face_x, self.face_y, self.img_x, self.img_y)

        def run_threaded(self,img_arr):
                self.img_arr = img_arr
                return self.out_img_arr, self.face_x, self.face_y, self.img_x, self.img_y
                        


        def shutdown(self):
                # indicate that the thread should be stopped
                self.on = False
                print('stoping face')
                
                #Clean up
                time.sleep(.5)
                #self.stream.close()
                #self.rawCapture.close()
                #self.camera.close()


