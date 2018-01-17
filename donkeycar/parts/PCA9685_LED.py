import time
import numpy as np
import Adafruit_PCA9685

#pwm = Adafruit_PCA9685.PCA9685()
#pwm.set_pwm_freq(60) 

class rgb_led:
    def __init__(self, red_channel = 5, green_channel = 6, blue_channel = 7,frequency=60):
        self.pwm = Adafruit_PCA9685.PCA9685()
        #self.pwm.set_pwm_freq(frequency)
        self.red_channel = red_channel
        self.green_channel = green_channel
        self.blue_channel = blue_channel
        #self.color=color
        self.led=(self.red_channel,self.green_channel,self.blue_channel)
        RGB_LED_set_color(self.pwm , self.led , (0,25,0)) 

    def set_color(self,color):
        RGB_LED_set_color(self.pwm , self.led , color)


    def run(self):
        pass


class turn_signal():
    def __init__(self, left_led=None, right_led=None):
        self.left_led = left_led
        self.right_led = right_led
         
 
    def run(self,angle):
        
        if angle > 0:
            
            self.right_led.set_color(((25 * abs(angle)),(25 * abs(angle)),0))
            self.left_led.set_color((0,0,0))

        if angle < 0:
            self.left_led.set_color(((25 * abs(angle)),(25 * abs(angle)),0))
            self.right_led.set_color((0,0,0))

        if angle == 0:
            self.left_led.set_color((0,0,0))
            self.right_led.set_color((0,0,0))

class status_indicator():
    def __init__(self, status_led=None):
        self.status_led = status_led

    def run(self,user_mode,recording):
        #print(str(user_mode))
        
        if recording:
            self.status_led.set_color((255,0,0))
        else:
            if user_mode == 'user':
                self.status_led.set_color((0,50,0))
            elif user_mode == 'local':
                self.status_led.set_color((0,0,255))
            elif user_mode == 'local_angle':
                self.status_led.set_color((255,0,255))


def main():
    #import Adafruit_PCA9685
    pwm = Adafruit_PCA9685.PCA9685()
    pwm.set_pwm_freq(60)
 
 
    # set up the LED pins.  (R,G,B)
    LEDS = (
        (5,6,7),
        (9,10,11),
        (13,14,15)
    )
    
    #initiales LEDS to purple
    for led in LEDS:
        RGB_LED_set_color(pwm,led,(20,0,50))    

    # set LEDs to RGB dim purple.. RED and GREEN scaled down
    #RGB_LED_set_color(pwm,LEDS[0],(255,0,255),scale=0.01)
    #RGB_LED_set_color(pwm,LEDS[1],(255,0,255),scale = 0.01)
    #RGB_LED_set_color(pwm,LEDS[2],(255,0,255),scale=0.01)

    #RGB_LED_set_color(pwm,LEDS[0],(20,0,50))
    #RGB_LED_set_color(pwm,LEDS[1],(20,0,50))
    #RGB_LED_set_color(pwm,LEDS[2],(20,0,50))
    

    ##time.sleep(3)
    #t = 1 
    #l = 255
    #pwm.set_pwm(11,0,4095)   
    ''' 
    while True:
        RGB_LED_set_color(pwm,LED_0,(l,0,0))
        time.sleep(t)

        RGB_LED_set_color(pwm,LED_0,(0,l,0))
        time.sleep(t)
        
        RGB_LED_set_color(pwm,LED_0,(0,0,l))
        time.sleep(t)

        RGB_LED_set_color(pwm,LED_0,(0,0,0))
        time.sleep(t)
 


        RGB_LED_set_color(pwm,LED_1,(l,0,0))
        time.sleep(t)

        RGB_LED_set_color(pwm,LED_1,(0,l,0))
 
        time.sleep(t)
        RGB_LED_set_color(pwm,LED_1,(0,0,l))
        time.sleep(t)
       
        RGB_LED_set_color(pwm,LED_1,(0,0,0))
        time.sleep(t)




        RGB_LED_set_color(pwm,LED_2,(l,0,0))
        time.sleep(t)

        RGB_LED_set_color(pwm,LED_2,(0,l,0))
        time.sleep(t)
 
        RGB_LED_set_color(pwm,LED_2,(0,0,l))
        time.sleep(t)
        
        RGB_LED_set_color(pwm,LED_2,(0,0,0))
        time.sleep(t)
        
        '''



def RGB_LED_set_color(driver , pins=(13,14,15) , RGB=(255,255,255) , CommonAnode=True, scale=1):
    
    # I only have Common Anode RGB leds, I THINK that Common Cathod LED's work differntly. 
    # To DO.  Add Common Cathode support
    if CommonAnode:
        #invert rgb input values so 255,0,0 input is RED and not the oposite.. 
        outRGB= (
            int(abs(((RGB[0] * scale) - 255 )*(16))),
            int(abs(((RGB[1] * scale) - 255 )*(16))),
            int(abs(((RGB[2] * scale) - 255 )*(16)))
            )

        #print('input: ' + str(RGB) + 'inveted' + str(outRGB))
    else:
        outRGB = RGB
        #for val in RGB:
        #for i in range(0 , RGB.count())
        #    print(str(RGB[i]))
            #invertedI =np.invert(np.array(val),dtype=np.int8)
            #print(invertedI)
            #RGB[i] = invertedI 
 
    # set the values for each R G B channel
    driver.set_pwm(pins[0],0,outRGB[0])
    driver.set_pwm(pins[1],0,outRGB[1])
    driver.set_pwm(pins[2],0,outRGB[2])


if __name__ == "__main__":
    main() 