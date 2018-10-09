import cv2
import numpy as np

class ImgGreyscale():

    def run(self, img_arr):
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        return img_arr

class ImgResize():
    def __init__(self, dimx=120,dimy=160):
        self.dimx=dimx
        self.dimy=dimy
        self.img_arr = np.zeros((dimx,dimy,3), np.uint8)
        
        
    def run(self, img_arr):
        resized = cv2.resize(img_arr,(160,120))
        return resized

class ImgDrawLine():

    def __init__(self, start=(0,40),end=(160,40),color=(255,0,0), width=5):
        self.start = start
        self.end = end
        self.color = color
        self.width = width
         
        
    def run(self, img_arr):
        #img_arr = cv2.line(img_arr,(0,40),(160,40),(255,0,0),5)
        img_arr = cv2.line(img_arr,self.start, self.end,self.color,self.width,cv2.LINE_AA)
        return img_arr
    
class ImgBoostBright():
        
    def __init__(self, value=30):
        self.value = value

    def run(self, img_arr):
        hsv = cv2.cvtColor(img_arr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - self.value
        v[v > lim] = 255
        v[v <= lim] += self.value

        final_hsv = cv2.merge((h, s, v))
        img_arr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img_arr
 

class ImgDrawArrow():

    def __init__(self, start=(0,40),end=(160,40),color=(255,0,0), width=5):
        self.start = start
        self.end = end
        self.color = color
        self.width = width
        
        
    def run(self, img_arr):
        #img_arr = cv2.line(img_arr,(0,40),(160,40),(255,0,0),5)
        img_arr = cv2.larrowedLine(img_arr,self.start, self.end,self.color,self.width,cv2.LINE_AA)
        return img_arr

class ImgPutInfo():

    def __init__(self, text=('text here'), org=(0,20),color=(255,255,255), width=1):
        self.org = org
        self.text = text
        self.color = color
        self.width = width
        
        
    def run(self, img_arr, throttle,angle):
        img_arr = cv2.putText(img_arr,'SPD: '+str(throttle), (1,15), cv2.FONT_HERSHEY_SIMPLEX,0.5, self.color,self.width, cv2.LINE_AA)
        img_arr = cv2.putText(img_arr,'ANG: '+str(angle), (1,35), cv2.FONT_HERSHEY_SIMPLEX,0.5, self.color,self.width, cv2.LINE_AA)

        return img_arr

class ImgPutText():

    def __init__(self, text=('text here'), org=(0,20),color=(255,255,255), size=0.4, width=1):
        self.org = org
        self.text = text
        self.color = color
        self.width = width
        self.size = size
        
        
    def run(self, img_arr, displaytext):
        img_arr = cv2.putText(img_arr, str(displaytext), (10,40), cv2.FONT_HERSHEY_SIMPLEX,self.size, self.color,self.width, cv2.LINE_AA)
        
        return img_arr

class ImgCanny():

    def __init__(self, low_threshold=60, high_threshold=110):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        
    def run(self, img_arr):
        return cv2.Canny(img_arr, 
                         self.low_threshold, 
                         self.high_threshold)

    

class ImgGaussianBlur():

    def __init__(self, kernal_size=5):
        self.kernal_size = kernal_size
        
    def run(self, img_arr):
        return cv2.GaussianBlur(img_arr, 
                                (self.kernel_size, self.kernel_size), 0)



class ImgCrop:
    """
    Crop an image to an area of interest. 
    """
    def __init__(self, top=0, bottom=0, left=0, right=0):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        
    def run(self, img_arr):
        height,width, _ = img_arr.shape
        img_arr = img_arr[self.top:height-self.bottom,
                          self.left: width-self.right]
        return img_arr
        


class ImgStack:
    """
    Stack N previous images into a single N channel image, after converting each to grayscale.
    The most recent image is the last channel, and pushes previous images towards the front.
    """
    def __init__(self, num_channels=3):
        self.img_arr = None
        self.num_channels = num_channels

    def rgb2gray(self, rgb):
        '''
        take a numpy rgb image return a new single channel image converted to greyscale
        '''
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        
    def run(self, img_arr):
        width, height, _ = img_arr.shape        
        gray = self.rgb2gray(img_arr)
        
        if self.img_arr is None:
            self.img_arr = np.zeros([width, height, self.num_channels], dtype=np.dtype('B'))

        for ch in range(self.num_channels - 1):
            self.img_arr[...,ch] = self.img_arr[...,ch+1]

        self.img_arr[...,self.num_channels - 1:] = np.reshape(gray, (width, height, 1))

        return self.img_arr

        
        
class Pipeline():
    def __init__(self, steps):
        self.steps = steps
    
    def run(self, val):
        for step in self.steps:
            f = step['f']
            args = step['args']
            kwargs = step['kwargs']
            
            val = f(val, *args, **kwargs)
        return val
    
