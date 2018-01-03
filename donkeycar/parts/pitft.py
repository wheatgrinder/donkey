#!/usr/bin/env python3

'''
donkey part to work with adafruitt PiTFT 3.5" LCD display.
https://learn.adafruit.com/adafruit-pitft-3-dot-5-touch-screen-for-raspberry-pi/displaying-images

classes

display text on display
So far I have only been able to find a methond to write to the display using the frame buffer.
(without installing x anyway which is really not necessary)

display image 

We will use opencv to create an image and display it on the screen
or we will display existing images on the screen
or we will show a short video on the screen.  (animaated gifts etc)

image size 480x330
 

'''
import os
import sys
import cv2
import numpy as np
import re
import time

 


class display_text():
    def __init__(self, port='/dev/fb1', text='text to show here'):
        self.port = port
        self.text = text
        self.prev_text = ''
        self.on = True


    def update(self):
        while self.on:
            
            #self.text = str(text)
            if not self.text == self.prev_text: # only update display if text is chnaged
                # = os.popen('vcgencmd measure_temp').readline()
                #print('display: '+ self.text)
                my_img = create_blank_image()
                my_img = add_text(my_img,self.text)
                my_img = display_image(my_img)
                self.prev_text = self.text
                time.sleep(2)
                
            
    def run_threaded(self,text):
        self.text = str(text)
        return self.prev_text

def create_blank_image():
    blank_image = np.zeros((330,480,3), np.uint8)
    #display_image_direct(blank_image)
    return blank_image
    
def display_image_direct(img_in):
    fbfd = open('/dev/fb1','w')
    fbfd.write(img_in)

def add_text(img_arr,text='text here',size=1,color=(255,255,255),width=1):
    img_arr = cv2.putText(img_arr, str(text), (20,100), cv2.FONT_HERSHEY_SIMPLEX,size, color,width, cv2.LINE_AA)
    return img_arr    

def open_image(img):
    image = cv2.imread(img,0)
    return image
     
 
def display_image(img_in):
    #write the imgage then display...
    image_file = cv2.imwrite(os.getcwd()+'/999.jpg', img_in)    
    img_handel = os.popen('sudo fbi -T 2 -d /dev/fb1 -noverbose -a ' + os.getcwd()+'/999.jpg > /dev/null 2>&1')
    return img_handel

def main():
    dir_path=os.getcwd()
    # my code here
    #result=os.popen('sudo fbi -T 2 -d /dev/fb1 -noverbose -a ' + dir_path + '/debug.jpg')
    #display_image(dir_path + '/debug.gif')
    my_img = create_blank_image()
    #my_img = open_image(os.getcwd()+'/dog.jpg')
    my_ip=os.popen('ip -4 addr show wlan0 | grep -oP \'(?<=inet\s)\d+(\.\d+){3}\'').readline()
    my_ip = re.sub('\?', '',my_ip) 
    text = 'View on Web at: ' + my_ip
    #text=os.popen('ip -4 addr show wlan0' ).readline()
    my_img = add_text(my_img,text)
    my_img = display_image(my_img)
    return
  
if __name__ == "__main__":
    main()
 

 