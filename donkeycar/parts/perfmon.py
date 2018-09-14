"""
Simple part to report some basic stats on the console every loop.

driveLoopTime = how long the drive loop took to get back around to runnign this part again.. 

coreTemp = reports the temperature of the pi for every loop  

throttled = returns the current system throttled status for every loop..  
:::::::WARNING::::

running the throttled monitor will really impact your drive loop time, it takes about 70 ms to complete so dont use should thred this part.. 
        if no throttoling has occured you should see "throttled=0x0"
        
        if throttoling has occured you will get a code like this throttled=50005
        0: under-voltage
        1: arm frequency capped
        2: currently throttled 
        16: under-voltage has occurred
        17: arm frequency capped has occurred
        18: throttling has occurred



usage:  add this part to  your car like so.. 

# import the part
from donkeycar.parts.perfmon import driveLoopTime
.......

#Initialize car
    V = dk.vehicle.Vehicle() 
.......
    loop_time = driveLoopTime() 
    V.add(loop_time,inputs = ['timein'], outputs = ['timein'])
.......


"""
import time
import os

class driveLoopTime():
    def __init__(self,timein =  time.time(),console=False):
        self.timein = timein
        self.console = console
        self.outtime = 0

    def run(self,timein):
        self.timein = timein
        returntime = time.time()
        
        if self.timein:
            self.elapsed_time = time.time() - self.timein
            self.outtime = "{0:.4f} ms".format (self.elapsed_time * 1000)
            if self.console:
                print('time in drive loop: ' + self.outtime) 

        return returntime, self.outtime

class coreTemp():
    #def __init__(self)

    def run(self):
        temp = os.popen('vcgencmd measure_temp').readline()
        print('core temp: ' + str(temp.rstrip()))
        return

class throttled():
    def update(self):
        while True:
            self.throttled_status = os.popen('vcgencmd get_throttled').readline()
            self.throttled_status = str(self.throttled_status.rstrip())
            time.sleep(10)
            #print(str(throttled_status.rstrip()))
        
    def run_threaded(self):
        return self.throttled_status
