"""
Pan and Tilt camera thingy
by wheatgrinder
"""

import time
#import numpy as np
import donkeycar as dk

class PCA9685:
    ''' 
    PWM motor controler using PCA9685 boards. 
    This is used for most RC Cars
    '''
    def __init__(self, channel, frequency=60):
        import Adafruit_PCA9685
        # Initialise the PCA9685 using the default address (0x40).
        self.pwm = Adafruit_PCA9685.PCA9685()
        self.pwm.set_pwm_freq(frequency)
        self.channel = channel

    def set_pulse(self, pulse):
        self.pwm.set_pwm(self.channel, 0, pulse) 

    def run(self, pulse):
        self.set_pulse(pulse)

class PWMPanning:
    """
    Wrapper over a PWM motor cotnroller to convert -1 to 1 throttle
    values to PWM pulses.
    PAN_CHANNEL = 2
    PAN_RIGHT_PWM = 100
    PAN_LEFT_PWM = 520
    PAN_CENTER_PWM = 295
    """
    LEFT_PAN = -1   
    RIGHT_PAN = 1  

    def __init__(self, controller=None,
                       left_pulse=520,
                       right_pulse=100,
                       zero_pulse=295):

        self.controller = controller
        self.left_pulse = left_pulse
        self.right_pulse = right_pulse
        self.zero_pulse = zero_pulse
        
        #send zero pulse to calibrate
        self.controller.set_pulse(self.zero_pulse)
        time.sleep(1)


    def run(self, pan):
        if pan > 0:
            pulse = dk.utils.map_range(pan,
                                    0, self.LEFT_PAN, 
                                    self.zero_pulse, self.left_pulse)
        else:
            pulse = dk.utils.map_range(pan,
                                    self.RIGHT_PAN, 0, 
                                    self.right_pulse, self.zero_pulse)

        self.controller.set_pulse(pulse)
        
    def shutdown(self):
        self.run(0) #stop vehicle

class PWMTilting:
    """
    Wrapper over a PWM motor cotnroller to convert -1 to 1 throttle
    values to PWM pulses.
    TILT_DOWN_PWM = 525
    TILT_CENTER_PWM = 260
    """
    MIN_TILT = -1  # furthest down look
    MAX_TILT =  1  # furthest look up

    def __init__(self, controller=None,
                       max_pulse=150,
                       min_pulse=525,
                       zero_pulse=260):

        self.controller = controller
        self.max_pulse = max_pulse
        self.min_pulse = min_pulse
        self.zero_pulse = zero_pulse
        
        #send zero pulse to calibrate
        self.controller.set_pulse(self.zero_pulse)
        time.sleep(1)


    def run(self, tilt):
        if tilt > 0:
            pulse = dk.utils.map_range(tilt,
                                    0, self.MAX_TILT, 
                                    self.zero_pulse, self.max_pulse)
        else:
            pulse = dk.utils.map_range(tilt,
                                    self.MIN_TILT, 0, 
                                    self.min_pulse, self.zero_pulse)

        self.controller.set_pulse(pulse)
        
    def shutdown(self):
        self.run(0) #stop vehicle


class PWMWagging:
    """
    Wrapper over a PWM motor cotnroller to convert -1 to 1 throttle
    values to PWM pulses.
    TAIL_CHANNEL = 15
    TAIL_UP_PWM = 300
    TAIL_DOWN_PWM = 100
    TAIL_CENTER = 200
    """
    MIN_WAG = -1  # furthest down look
    MAX_WAG =  1  # furthest look up

    def __init__(self, controller=None,
                       max_pulse=150,
                       min_pulse=525,
                       zero_pulse=260):

        self.controller = controller
        self.max_pulse = max_pulse
        self.min_pulse = min_pulse
        self.zero_pulse = zero_pulse
        
        #send zero pulse to calibrate
        self.controller.set_pulse(self.zero_pulse)
        time.sleep(1)


    def run(self, tilt):
        if wag > 0:
            pulse = dk.utils.map_range(wag,
                                    0, self.MAX_WAG, 
                                    self.zero_pulse, self.max_pulse)
        else:
            pulse = dk.utils.map_range(wag,
                                    self.MIN_WAG, 0, 
                                    self.min_pulse, self.zero_pulse)

        self.controller.set_pulse(pulse)
        
    def shutdown(self):
        self.run(0) #stop vehicle