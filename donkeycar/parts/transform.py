# -*- coding: utf-8 -*-

import time
import numpy as np
import donkeycar as dk
#import math

class Lambda:
    """
    Wraps a function into a donkey part.
    """
    def __init__(self, f):
        """
        Accepts the function to use.
        """
        self.f = f

    def run(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def shutdown(self):
        return

class LookAt:  
    """
    I moved this code into the NCS part.. dont work on it here. 
     inputs screen coordinates of the detected object (target) and (current pan, current tilt)  
    and returns new_pan and new_tilt value twards the center of the detected ojbect target with some slop
    """
    def __init__(self, screen_center=(224,224), slop=50, p=0, i=0, d=0,look_for='person', debug=False):
        self.target_center =(0,0)
        self.screen_center = screen_center
        self.debug = debug
        self.slop = slop
        self.look_for = look_for
        # initialize gains
        self.Kp = p
        self.Ki = i
        self.Kd = d

        # The value the controller is trying to get the system to achieve.
        self.target = 0

        # initialize delta t variables
        self.prev_tm = time.time()
        self.prev_feedback = 0
        self.error = None

        # initialize the output
        self.alpha = 0
        self.diff = 0
        self.change_amt = 0


    def run(self, obj_list,  pan_in, tilt_in):
        #self.target_center=target_center
        #(float(target_center[0]),float(target_center[1]))
        self.obj_list = obj_list
       
        
        self.pan_in = float(pan_in)
        self.tilt_in = float(tilt_in)
        self.pan = float(pan_in)
        self.tilt = float(tilt_in)
        
        """
        network_classifications = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
        
        iterate over object in and find the first instance of the "look_for" as
        val = default_val
        for x in some_list:
            if match(x):
            val = x
            break

            #['person', 209.45312, 229.90625, 310.50378, 406.15082, 0.37283849716186523]
        """
        
        for obj in self.obj_list:
            print(obj)
            print(obj[0])
            if self.look_for in obj:
                self.target_center=(obj[1],obj[2])
                break
        
        


        change_amt = 0.01
        if self.screen_center and self.target_center:
            #self.diff = int(abs((self.target_center[0] - self.screen_center[0]) + (self.target_center[1] - self.screen_center[1])))
            
            sc=np.array(self.screen_center)
            tc=np.array(self.target_center)

            td = tc-sc
            self.diff = np.sqrt(td.dot(td))
            #def scale_number(unscaled, to_min, to_max, from_min, from_max):
            #self.diff = scale_number(self.diff,0.002,0.1,0,200)
            #(to_max-to_min)*(unscaled-from_min)/(from_max-from_min)+to_min
            
            self.change_amt = self.diff * 0.0002
           
            #self.change_amt = (0.1-0.001)*(self.diff-0)/(50-0)+0.001
            #def map_range(x, X_min, X_max, Y_min, Y_max):
            #self.change_amt = dk.util.data.map_range(self.diff,0,150,0.001,0.1)
            #self.change_amt = change_amt

            #if   self.diff > self.slop: 
            
            if self.pan > -1 and self.pan < 1:  
                
                if self.target_center[0] > self.screen_center[0]:
                    self.pan = self.pan + self.change_amt
                else:
                    self.pan = self.pan - self.change_amt

            if self.tilt > -1 and self.tilt < 1: 
                if self.target_center[1] < self.screen_center[1]:
                    self.tilt = self.tilt + self.change_amt
                else:
                    self.tilt = self.tilt - self.change_amt

        if self.debug:
            print('diff:', self.diff, " change_amt: ", self.change_amt, " screen center:" , self.screen_center," target center:" , self.target_center, 
            " pan_in:", self.pan_in, " tilt_in:", self.tilt_in, 
            " pan:", self.pan, " tilt:", self.tilt)

        return self.pan, self.tilt

   




class PIDController:
    """ Performs a PID computation and returns a control value.
        This is based on the elapsed time (dt) and the current value of the process variable
        (i.e. the thing we're measuring and trying to change).
        https://github.com/chrisspen/pid_controller/blob/master/pid_controller/pid.py
    """

    def __init__(self, p=0, i=0, d=0, debug=False):

        # initialize gains
        self.Kp = p
        self.Ki = i
        self.Kd = d

        # The value the controller is trying to get the system to achieve.
        self.target = 0

        # initialize delta t variables
        self.prev_tm = time.time()
        self.prev_feedback = 0
        self.error = None

        # initialize the output
        self.alpha = 0

        # debug flag (set to True for console output)
        self.debug = debug

    def run(self, target_value, feedback):
        curr_tm = time.time()

        self.target = target_value
        error = self.error = self.target - feedback

        # Calculate time differential.
        dt = curr_tm - self.prev_tm

        # Initialize output variable.
        curr_alpha = 0

        # Add proportional component.
        curr_alpha += self.Kp * error

        # Add integral component.
        curr_alpha += self.Ki * (error * dt)

        # Add differential component (avoiding divide-by-zero).
        if dt > 0:
            curr_alpha += self.Kd * ((feedback - self.prev_feedback) / float(dt))

        # Maintain memory for next loop.
        self.prev_tm = curr_tm
        self.prev_feedback = feedback

        # Update the output
        self.alpha = curr_alpha

        if (self.debug):
            print('PID target value:', round(target_value, 4))
            print('PID feedback value:', round(feedback, 4))
            print('PID output:', round(curr_alpha, 4))

        return curr_alpha
