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




class display_text():
    def __init__(self, port='/dev/fb1', text='text to show here'):
        self.port = port
        self.text = text
        self.prev_text = ''
        self.on = True


    def update(self):
        if self.on:
            if not self.text == self.prev_text: # only update display if text is chnaged
                # = os.popen('vcgencmd measure_temp').readline()
                print('update')
            
    def run_threaded(self):
        return 

def create_blank_image():
    blank_image = np.zeros((330,480,3), np.uint8)
    display_image_direct(blank_image)    
    
def display_image_direct(img_in):
    fbfd = open('/dev/fb1','w')
    fbfd.write(img_in)
    
    
    

def display_image(img_in):
    img_handel = os.popen('sudo fbi -T 2 -d /dev/fb1 -noverbose -a ' + img_in)
    return img_handel
    

def main():
    dir_path=os.getcwd()
    # my code here
    #result=os.popen('sudo fbi -T 2 -d /dev/fb1 -noverbose -a ' + dir_path + '/debug.jpg')
    #display_image(dir_path + '/debug.jpg')
    create_blank_image()


if __name__ == "__main__":
    main()


