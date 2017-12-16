#! /usr/bin/env python3



import sys
import numpy as np
import cv2
import time
import csv
import os
import sys

class googlenet():
        def __init__(self, BASE_DIR = '/home/pi/d2/models/ncs_data/', NETWORK_IMAGE_WIDTH = 224, NETWORK_IMAGE_HEIGHT = 224):

                from mvnc import mvncapi as mvnc

                
                # initialize the ncs and network params
                self.NETWORK_IMAGE_WIDTH = NETWORK_IMAGE_WIDTH
                self.NETWORK_IMAGE_HEIGHT = NETWORK_IMAGE_HEIGHT
                self.BASE_DIR = BASE_DIR
        
                # initialize the frame and the variable used to indicate
                # if the thread should be stopped
                #self.frame = None
                self.on = True
                

                # ***************************************************************
                # get labels
                # ***************************************************************
                labels_file= self.BASE_DIR+'ilsvrc12/synset_words.txt'
                self.labels=np.loadtxt(labels_file,str,delimiter='\t')

                # ***************************************************************
                # configure the NCS
                # ***************************************************************
                mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
                
                print('Starting Neural Compute Stick')

                # Set logging level and initialize/open the first NCS we find
                mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 0)
                devices = mvnc.EnumerateDevices()
                if len(devices) == 0:
                        print('No devices found')
                        return 1
                device = mvnc.Device(devices[0])
                try:
                        device.OpenDevice()
                except:
                        print("Error - could not open the NCS device")

                print("NCS Device Opened normally. - warming NCS")

                network_blob=self.BASE_DIR+'googlenet_graph'

                #Load blob
                with open(network_blob, mode='rb') as f:
                        blob = f.read()
                        self.graph = device.AllocateGraph(blob)

                        self.ilsvrc_mean = np.load(self.BASE_DIR+'ilsvrc12/ilsvrc_2012_mean.npy').mean(1).mean(1) #loading the mean file

                print("NCS Device Graph is loaded, ready to classify")
                time.sleep(2)
                self.out_data='none'
                #self.img_arr= (cv2.Mat image(320, 240, CV_8UC3, Scalar(0,0,0)))
                self.img_arr = np.zeros((1,1,3), np.uint8)
                self.hldCnt = 0;        

                
        def update(self):
                
                while self.on:
                        
                        t = time.time()        
                        
                        # do complicated stuff here..
                        # ***************************************************************
                        # Load the image
                        # ***************************************************************
                        #self.img_arr='cam/image_array'

                        #img = cv2.imread(EXAMPLES_BASE_DIR+'data/images/nps_electric_guitar.png')
                        dim = (self.NETWORK_IMAGE_WIDTH, self.NETWORK_IMAGE_HEIGHT)
                        img = self.img_arr
                        img=cv2.resize(img,dim)
                        img = img.astype(np.float32)
                        img[:,:,0] = (img[:,:,0] - self.ilsvrc_mean[0])
                        img[:,:,1] = (img[:,:,1] - self.ilsvrc_mean[1])
                        img[:,:,2] = (img[:,:,2] - self.ilsvrc_mean[2])


                        # ***************************************************************
                        # Send the image to the NCS
                        # ***************************************************************
                        self.graph.LoadTensor(img.astype(np.float16), 'user object')


                        # ***************************************************************
                        # Get the result from the NCS
                        # ***************************************************************
                        output, userobj = self.graph.GetResult()

                        # ***************************************************************
                        # Print the results of the inference form the NCS
                        # ***************************************************************
                        
                        order = output.argsort()[::-1][:6]
                        #self.out_data='';

                        #print('\n------- predictions --------')
                        #for i in range(0,1):
                        #        print ('prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + self.labels[order[i]] + '  label index is: ' + str(order[i]) )
                        #
                        #i=0        
                        #print ('prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + self.labels[order[i]] + '  label index is: ' + str(order[i]) )
                        #      
                                #self.out_data += 'prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + self.labels[order[i]] + '  label index is: ' + str(order[i]) 
                                #self.out_data = self.labels[order[i]] +':'+ str(output[order[i]])

                        i=0
                        #if output[order[i]] > 0.2:
                        #        self.out_data = self.labels[order[i]] +':'+ str(output[order[i]])      
                                #print(self.out_data)

                        # pares the labels.. only want the first real string..
                        # typical line looks like this.. I want jut "tailed frog"
                        # n01644900 tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui
                        # split into comma list...
                        elapsed_time = time.time() - t
                        if output[order[0]] > 0.1:
                                outLable = self.labels[order[0]].split(",") 
                                outLable = outLable[0].split(" ",1)  #split at first space
                                outConfidence = "{0:.0f}%".format( output[order[0]] * 100)
                                outTime = "{0:.0f}ms".format (elapsed_time*1000)
                                
                                self.out_data = outConfidence +' : '+ outLable[1] + " : " + outTime
                                
                        else:
                               self.out_data = ''        

                        #print(self.out_data)
                        

                        #return self.out_data
                        # simulate a long time..         
                        #time.sleep(1)

                
        def runXXX(self,img_arr):
                # keep looping infinitely until the thread is stopped
                self.img_arr = img_arr
                while self.on:
                        #self.out_data = self.run(self.img_arr)        
                        # do stuff
                        
                        if not self.on:
                                break
                return self.out_data

        def run_threaded(self,img_arr):
                self.img_arr = img_arr
                return self.out_data
                


        def shutdown(self):
                # indicate that the thread should be stopped
                self.on = False
                print('stoping ncs')
                
                #Clean up
                graph.DeallocateGraph()
                device.CloseDevice()
                time.sleep(.5)
                #self.stream.close()
                #self.rawCapture.close()
                #self.camera.close()
