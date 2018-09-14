#! /usr/bin/env python3
'''

the purpose of this part is to take inputs from driving inference and the tinyyolo results and affect the vehicles power and stearing.
it shoudl be setup in the manage.py just prior to the part that sets the angle and throttle.  

For example a person in "front" of the car should result in the car stoping and waiting for the person to move. This should overrite
any and all other behaviours from the networks.  Other objects may result in nudging the stearing towards or away from something by adding some
value to the driving inference.. 

from manage.py...

# add govenor part here.  Governer has overridding power when drive mode is pilot
    peeps = break_for('person')
    V.add(peeps, inputs=['user/mode','angle', 'throttle','ncs/found_objs'], outputs=['angle','throttle'])


    V.add(steering, inputs=['angle'])
    V.add(throttle, inputs=['throttle'])




These are the objects that are detected.. you can setup your braek_for to use anyof these.. 

network_classifications = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                                        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                                        "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]


named inputs are drive_mode, angel, throttle, found_obj

    V.add(steering, inputs=['angle'])
    V.add(throttle, inputs=['throttle'])

'''

import sys
#import numpy as np
#import cv2
#import time
#import csv
#import os
 
class break_for:

    def __init__(self, break_for_obj='person', confidence=.3, wait_for_frames=60):
        self.break_for_obj = break_for_obj
        self.confidence = confidence
        self.wait_for_frames = wait_for_frames
        self.frame_cnt = self.wait_for_frames
        self.obj_in_path = False
 

    def run(self,mode,angle, throttle, found_objs):
        #print(mode)
        self.angle = angle
        self.throttle = throttle
        
        
        #check if timer is up
        if self.obj_in_path and self.frame_cnt <= self.wait_for_frames:
            #timer is still in effect
            #increment timer by 1
            self.obj_in_path = True
            self.throttle = 0  # override input
            self.frame_cnt = self.frame_cnt + 1
            print(self.break_for_obj + ' in path. waiting:' + str(self.frame_cnt))
        else:
            #frame counter is up, back to normal procesing.. 
            self.obj_in_path = False
            self.frame_cnt = 0  #set the counter back to 0 every frame when no obj detected.. 
            self.throttle = self.throttle #pass the input
            #print('timer over, start checking again')
           
            #do the check for obj
            if mode=='local':   # this car is contorled by the nn.  in local_angle mode the user is still controlling start stop.. 

                #print('car is under local control')
                
                for obj_index in range(len(found_objs)):
                    if str(found_objs[obj_index][0]) == self.break_for_obj and found_objs[obj_index][5] >= self.confidence:
                        self.obj_in_path = True
                        self.frame_cnt = 0
                        
                    '''
                    thing =  found_objs[obj_index][0]
                    center_x = int(found_objs[obj_index][1])
                    center_y = int(found_objs[obj_index][2])
                    half_width = int(found_objs[obj_index][3])//2
                    half_height = int(found_objs[obj_index][4])//2
                    #confidence = ' : %.2f' % filtered_objects[obj_index][5]
                    confidence =  found_objs[obj_index][5]


                    # calculate box (left, top) and (right, bottom) coordinates
                    box_left = max(center_x - half_width, 0)
                    box_top = max(center_y - half_height, 0)
                    #box_right = min(center_x + half_width, source_image_width)
                    #box_bottom = min(center_y + half_height, source_image_height)
                    print(str(thing) + ':' + str(confidence) )
                    '''
        print('obj in path:' + str(self.obj_in_path) )
        return self.angle, self.throttle