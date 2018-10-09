#! /usr/bin/env python3
'''
To use this part you will need to have a pretrained model, commonaly called a graph file in NCS speak
Training and generating graph files is beyond the scope of this little bit of help

You will also need th movidus neural compute stick libraries installed on your PI... call this an API mode install (very light weight)
Basic instructions to install NCS

                donkey_v22.img
                Plug in ethernet cable
                connected ssh via ethernet
                user pi
                pwd asdfasdf

                sudo apt-get install libusb-1.0-0-dev
                mkdir workspace
                cd workspace/
                
                git clone https://github.com/movidius/ncsdk
                
                cd ncsdk/api/src/
                sudo make install

                cd ~/workspace/ncsdk/examples/apps/hello_ncs_py
                python3 hello_ncs.py;
                Hello NCS! Device opened normally.  
                Goodbye NCS! Device closed normally.

once the NCS is installed an you have those results showing its working
Get the trained graph files from the github https://github.com/wheatgrinder/donkeycar_cars/tree/master/d2/models/ncs_data
you only need the graph file if your running the yolo, for googlenet you will also need the synset_words.txt and ilsvrc_2012_mean.npy files

..... or setup the NCS dev environment on your pc and create the graph etc..  

Place the files in your cars "models" directory
Add the ncs part to your cars manage.py file...


        #NCS PARTS
        from donkeycar.parts.ncs import googlenet
        from donkeycar.parts.ncs import tinyyolo


    ncs_ty = tinyyolo(basedir = cfg.MODELS_PATH, draw_on_img = True, probability_threshold = 0.1)
    V.add(ncs_ty, inputs=['cam/image_array'],outputs=['ncs/image_array','ncs/found_objs'],threaded=True)

You can have the data without affecting the image by setting "draw_on_img = False.  
You can use the data in another part and do soemthing with it see my "goveneor.py" part... 


the googlenet implementation outputs data and does not affect the image. you can do something useful withi the 'classification' output  

        ncs_gn = googlenet()
        ncs_gn.base_dir = '~/cars/d2/models/ncs_data/'
        V.add(ncs_gn, inputs=['cam/image_array'],outputs=['classificaiton'],threaded=True)


most this code is from the donkey project and the intel movidus ncazoo below.. there is some license stuff.. read it
'''


import sys
import numpy as np
import cv2
import time
import csv
import os


class googlenet():
        def __init__(self, basedir, NETWORK_IMAGE_WIDTH = 224, NETWORK_IMAGE_HEIGHT = 224, draw_on_img=True, probability_threshold = 0.2, debug=False):
                
                from mvnc import mvncapi as mvnc
                # initialize the ncs and network params
                self.NETWORK_IMAGE_WIDTH = NETWORK_IMAGE_WIDTH
                self.NETWORK_IMAGE_HEIGHT = NETWORK_IMAGE_HEIGHT
                self.BASE_DIR = basedir+'/'
                self.probability_threshold = probability_threshold
                self.on = True
                self.draw_on_img = draw_on_img
                self.debug = debug
                

                # ***************************************************************
                # get labels
                # ***************************************************************
                labels_file= self.BASE_DIR+'synset_words.txt'
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

                        self.ilsvrc_mean = np.load(self.BASE_DIR+'ilsvrc_2012_mean.npy').mean(1).mean(1) #loading the mean file

                print("NCS Device Graph is loaded, ready to classify")



                time.sleep(2)
                self.out_data='none'
                 # we need an image to start or 
                self.img_arr = np.zeros((1,1,3), np.uint8)
                self.out_img_arr = np.zeros((1,1,3), np.uint8)
                self.hldCnt = 0;        

                        
        def update(self):
                        
                while self.on:

                        #self.out_img_arr = self.img_arr

                        t = time.time()        
                        
                        # do complicated stuff here..
                        # ***************************************************************
                        # Load the image
                        # ***************************************************************
                        #self.img_arr='cam/image_array'

                        #img = cv2.imread(EXAMPLES_BASE_DIR+'data/images/nps_electric_guitar.png')
                        dim = (self.NETWORK_IMAGE_WIDTH, self.NETWORK_IMAGE_HEIGHT)
                        #if self.debug:
                        #        img = cv2.imread(self.BASE_DIR+'debug.jpg')
                        #else:
                        img = self.out_img_arr 
                        
                        
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
                        #for i in range(0,6):
                        #        print ('prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + self.labels[order[i]] + '  label index is: ' + str(order[i]) )
                        
                        # pares the labels.. only want the first real string..
                        # typical line looks like this.. I want jut "tailed frog"
                        # n01644900 tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui
                        # split into comma list...
                        elapsed_time = time.time() - t
                        if output[order[0]] >= self.probability_threshold:
                                outLable = self.labels[order[0]].split(",") 
                                outLable = outLable[0].split(" ",1)  #split at first space
                                outConfidence = "{0:.0f}%".format( output[order[0]] * 100)
                                outTime = "{0:.0f}ms".format (elapsed_time*1000)
                                
                                #self.out_data = outConfidence +' : '+ outLable[1] + " : " + outTime
                                self.out_data = (outConfidence, outLable[1], outTime)
                                #self.out_data = order
                        else:
                                self.out_data = ('','',outTime)        

                        #print(self.out_data)
                        if self.draw_on_img:
                                self.out_img_arr = self.display_in_gui(self.img_arr, self.out_data,elapsed_time)
                        else:
                                self.out_img_arr = self.img_arr
                        

                        #return self.out_data
                        # simulate a long time..         
                        #time.sleep(1)

        def display_in_gui(self,source_image, data_in, elapsed_time):
                display_image = source_image.copy()
                #source_image_width = source_image.shape[1]
                #source_image_height = source_image.shape[0]        
                display_image = cv2.putText(source_image,str(data_in[0]) + ' : ' + str(data_in[1]) +' : ' + str(data_in[2]), (1,15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255),1, cv2.LINE_AA)
                #display_image = cv2.putText(self.out_img_arr,'Object: '+str(data_in[1]), (1,35), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255),1, cv2.LINE_AA)
                #print(data_in[0])
                #self.out_img_arr = display_image
                return display_image


        def run_threaded(self,img_arr):
                self.img_arr = img_arr
                return self.out_img_arr, self.out_data
                        


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


class inception():
        def __init__(self, basedir, NETWORK_IMAGE_WIDTH = 224, NETWORK_IMAGE_HEIGHT = 224, draw_on_img=True, probability_threshold = 0.2,debug=False):
                
                from mvnc import mvncapi as mvnc
                # initialize the ncs and network params
                self.NETWORK_IMAGE_WIDTH = NETWORK_IMAGE_WIDTH
                self.NETWORK_IMAGE_HEIGHT = NETWORK_IMAGE_HEIGHT
                self.BASE_DIR = basedir+'/'
                self.probability_threshold = probability_threshold
                self.on = True
                self.draw_on_img = draw_on_img
                self.debug = debug
                                
                # ***************************************************************
                # get labels
                # ***************************************************************
                labels_file= self.BASE_DIR+'synset_words.txt'
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

                network_blob=self.BASE_DIR+'inception_v3_graph'

                #Load blob
                with open(network_blob, mode='rb') as f:
                        blob = f.read()
                        self.graph = device.AllocateGraph(blob)

                        self.ilsvrc_mean = np.load(self.BASE_DIR+'ilsvrc_2012_mean.npy').mean(1).mean(1) #loading the mean file

                print("NCS Device Graph is loaded, ready to classify")



                time.sleep(2)
                self.out_data='none'
                 # we need an image to start or 
                self.img_arr = np.zeros((1,1,3), np.uint8)
                self.out_img_arr = np.zeros((1,1,3), np.uint8)
                self.hldCnt = 0;        

                        
        def update(self):
                        
                while self.on:

                        #self.out_img_arr = self.img_arr

                        t = time.time()        
                        
                        # do complicated stuff here..
                        # ***************************************************************
                        # Load the image
                        # ***************************************************************
                        #self.img_arr='cam/image_array'

                        #img = cv2.imread(EXAMPLES_BASE_DIR+'data/images/nps_electric_guitar.png')
                        dim = (self.NETWORK_IMAGE_WIDTH, self.NETWORK_IMAGE_HEIGHT)
                        #img = self.out_img_arr
                        #if self.debug:
                        #        img = self.BASE_DIR+'debug.jpg'
                        #else:
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
                        #for i in range(0,6):
                        #        print ('prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + self.labels[order[i]] + '  label index is: ' + str(order[i]) )
                        
                        # pares the labels.. only want the first real string..
                        # typical line looks like this.. I want jut "tailed frog"
                        # n01644900 tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui
                        # split into comma list...
                        elapsed_time = time.time() - t
                        if output[order[0]] >= self.probability_threshold:
                                outLable = self.labels[order[0]].split(",") 
                                outLable = outLable[0].split(" ",1)  #split at first space
                                outConfidence = "{0:.0f}%".format( output[order[0]] * 100)
                                outTime = "{0:.0f}ms".format (elapsed_time*1000)
                                
                                #self.out_data = outConfidence +' : '+ outLable[1] + " : " + outTime
                                self.out_data = (outConfidence, outLable[1], outTime)
                                #self.out_data = order
                        else:
                                self.out_data = ('','',outTime)        

                        #print(self.out_data)
                        if self.draw_on_img:
                                self.out_img_arr = self.display_in_gui(self.img_arr, self.out_data,elapsed_time)
                        else:
                                self.out_img_arr = self.img_arr
                        

                        #return self.out_data
                        # simulate a long time..         
                        #time.sleep(1)

        def display_in_gui(self,source_image, data_in, elapsed_time):
                display_image = source_image.copy()
                #source_image_width = source_image.shape[1]
                #source_image_height = source_image.shape[0]        
                display_image = cv2.putText(source_image,str(data_in[0]) + ' : ' + str(data_in[1]) +' : ' + str(data_in[2]), (1,15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255),1, cv2.LINE_AA)
                #display_image = cv2.putText(self.out_img_arr,'Object: '+str(data_in[1]), (1,35), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255),1, cv2.LINE_AA)
                #print(data_in[0])
                #self.out_img_arr = display_image
                return display_image


        def run_threaded(self,img_arr):
                #if debug:
                self.img_arr = img_arr

                return self.out_img_arr, self.out_data
                        


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


class tinyyolo():
        
        def __init__(self, basedir , NETWORK_IMAGE_WIDTH = 448, NETWORK_IMAGE_HEIGHT = 448, draw_on_img=True, probability_threshold = 0.07,debug=False):
                
                from mvnc import mvncapi as mvnc
                # initialize the ncs and network params
                self.NETWORK_IMAGE_WIDTH = NETWORK_IMAGE_WIDTH
                self.NETWORK_IMAGE_HEIGHT = NETWORK_IMAGE_HEIGHT
                self.BASE_DIR = basedir+'/'
                self.draw_on_img = draw_on_img
                self.debug = debug

                self.on = True
                # only report ojbects with probabilities greater than this
                self.probability_threshold = probability_threshold

                self.first_obj_x = 0
                self.first_obj_y = 0
                self.center_screen_x = 0
                self.center_screen_y = 0
                #self.out_filtered_objs =[]
                # ***************************************************************
                # get labels
                # ***************************************************************
                #labels_file= self.BASE_DIR+'ilsvrc12/synset_words.txt'
                #self.labels=np.loadtxt(labels_file,str,delimiter='\t')

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

                tiny_yolo_graph_file=self.BASE_DIR+'tiny_yolo_graph'
                #tiny_yolo_graph_file= '/home/pi/cars/d2/models/ncs_data/tiny_yolo_graph'

                #Load blob
                 #Load graph from disk and allocate graph via API
                with open(tiny_yolo_graph_file, mode='rb') as f:
                        graph_from_disk = f.read()

                self.graph = device.AllocateGraph(graph_from_disk)

                #with open(network_blob, mode='rb') as f:
                #        blob = f.read()
                #        self.graph = device.AllocateGraph(blob)

                        #self.ilsvrc_mean = np.load(self.BASE_DIR+'ilsvrc12/ilsvrc_2012_mean.npy').mean(1).mean(1) #loading the mean file

                                # the 20 classes this network was trained on
                
                self.network_classifications = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                                        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                                        "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
                
                
                #self.network_classifications = ["car", "cat", "dog", "person" ]




                print("NCS Device Graph is loaded, ready to classify")
                time.sleep(2)
                self.out_data='none'
                # we need an image to start or 
                self.img_arr = np.zeros((1,1,3), np.uint8)
                self.out_img_arr = np.zeros((1,1,3), np.uint8)
                #self.hldCnt = 0;        


        def filter_objects(self,inference_result, input_image_width, input_image_height):
                
                # the raw number of floats returned from the inference (GetResult())
                num_inference_results = len(inference_result)
                '''
                # the 20 classes this network was trained on
                network_classifications = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                                        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                                        "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

                # only keep boxes with probabilities greater than this
                probability_threshold = 0.07
                '''

                num_classifications = len(self.network_classifications) # should be 20
                grid_size = 7 # the image is a 7x7 grid.  Each box in the grid is 64x64 pixels
                boxes_per_grid_cell = 2 # the number of boxes returned for each grid cell

                # grid_size is 7 (grid is 7x7)
                # num classifications is 20
                # boxes per grid cell is 2
                all_probabilities = np.zeros((grid_size, grid_size, boxes_per_grid_cell, num_classifications))

                # classification_probabilities  contains a probability for each classification for
                # each 64x64 pixel square of the grid.  The source image contains
                # 7x7 of these 64x64 pixel squares and there are 20 possible classifications
                classification_probabilities = \
                        np.reshape(inference_result[0:980], (grid_size, grid_size, num_classifications))
                num_of_class_probs = len(classification_probabilities)

                # The probability scale factor for each box
                box_prob_scale_factor = np.reshape(inference_result[980:1078], (grid_size, grid_size, boxes_per_grid_cell))

                # get the boxes from the results and adjust to be pixel units
                all_boxes = np.reshape(inference_result[1078:], (grid_size, grid_size, boxes_per_grid_cell, 4))
                self.boxes_to_pixel_units(all_boxes, input_image_width, input_image_height, grid_size)

                # adjust the probabilities with the scaling factor
                for box_index in range(boxes_per_grid_cell): # loop over boxes
                        for class_index in range(num_classifications): # loop over classifications
                                all_probabilities[:,:,box_index,class_index] = np.multiply(classification_probabilities[:,:,class_index],box_prob_scale_factor[:,:,box_index])


                probability_threshold_mask = np.array(all_probabilities>=self.probability_threshold, dtype='bool')
                box_threshold_mask = np.nonzero(probability_threshold_mask)
                boxes_above_threshold = all_boxes[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
                classifications_for_boxes_above = np.argmax(all_probabilities,axis=3)[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
                probabilities_above_threshold = all_probabilities[probability_threshold_mask]

                # sort the boxes from highest probability to lowest and then
                # sort the probabilities and classifications to match
                argsort = np.array(np.argsort(probabilities_above_threshold))[::-1]
                boxes_above_threshold = boxes_above_threshold[argsort]
                classifications_for_boxes_above = classifications_for_boxes_above[argsort]
                probabilities_above_threshold = probabilities_above_threshold[argsort]


                # get mask for boxes that seem to be the same object
                duplicate_box_mask = self.get_duplicate_box_mask(boxes_above_threshold)

                # update the boxes, probabilities and classifications removing duplicates.
                boxes_above_threshold = boxes_above_threshold[duplicate_box_mask]
                classifications_for_boxes_above = classifications_for_boxes_above[duplicate_box_mask]
                probabilities_above_threshold = probabilities_above_threshold[duplicate_box_mask]

                classes_boxes_and_probs = []
                for i in range(len(boxes_above_threshold)):
                        classes_boxes_and_probs.append([self.network_classifications[classifications_for_boxes_above[i]],boxes_above_threshold[i][0],boxes_above_threshold[i][1],boxes_above_threshold[i][2],boxes_above_threshold[i][3],probabilities_above_threshold[i]])

                return classes_boxes_and_probs

                # creates a mask to remove duplicate objects (boxes) and their related probabilities and classifications
                # that should be considered the same object.  This is determined by how similar the boxes are
                # based on the intersection-over-union metric.
                # box_list is as list of boxes (4 floats for centerX, centerY and Length and Width)
        def get_duplicate_box_mask(self,box_list):
                # The intersection-over-union threshold to use when determining duplicates.
                # objects/boxes found that are over this threshold will be
                # considered the same object
                max_iou = 0.35

                box_mask = np.ones(len(box_list))

                for i in range(len(box_list)):
                        if box_mask[i] == 0: continue
                        for j in range(i + 1, len(box_list)):
                                if self.get_intersection_over_union(box_list[i], box_list[j]) > max_iou:
                                        box_mask[j] = 0.0

                filter_iou_mask = np.array(box_mask > 0.0, dtype='bool')
                return filter_iou_mask

                # Converts the boxes in box list to pixel units
                # assumes box_list is the output from the box output from
                # the tiny yolo network and is [grid_size x grid_size x 2 x 4].
        def boxes_to_pixel_units(self,box_list, image_width, image_height, grid_size):

                # number of boxes per grid cell
                boxes_per_cell = 2

                # setup some offset values to map boxes to pixels
                # box_offset will be [[ [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]] ...repeated for 7 ]
                box_offset = np.transpose(np.reshape(np.array([np.arange(grid_size)]*(grid_size*2)),(boxes_per_cell,grid_size, grid_size)),(1,2,0))

                # adjust the box center
                box_list[:,:,:,0] += box_offset
                box_list[:,:,:,1] += np.transpose(box_offset,(1,0,2))
                box_list[:,:,:,0:2] = box_list[:,:,:,0:2] / (grid_size * 1.0)

                # adjust the lengths and widths
                box_list[:,:,:,2] = np.multiply(box_list[:,:,:,2],box_list[:,:,:,2])
                box_list[:,:,:,3] = np.multiply(box_list[:,:,:,3],box_list[:,:,:,3])

                #scale the boxes to the image size in pixels
                box_list[:,:,:,0] *= image_width
                box_list[:,:,:,1] *= image_height
                box_list[:,:,:,2] *= image_width
                box_list[:,:,:,3] *= image_height


                # Evaluate the intersection-over-union for two boxes
                # The intersection-over-union metric determines how close
                # two boxes are to being the same box.  The closer the boxes
                # are to being the same, the closer the metric will be to 1.0
                # box_1 and box_2 are arrays of 4 numbers which are the (x, y)
                # points that define the center of the box and the length and width of
                # the box.
                # Returns the intersection-over-union (between 0.0 and 1.0)
                # for the two boxes specified.
        def get_intersection_over_union(self,box_1, box_2):

                # one diminsion of the intersecting box
                intersection_dim_1 = min(box_1[0]+0.5*box_1[2],box_2[0]+0.5*box_2[2])-\
                                        max(box_1[0]-0.5*box_1[2],box_2[0]-0.5*box_2[2])

                # the other dimension of the intersecting box
                intersection_dim_2 = min(box_1[1]+0.5*box_1[3],box_2[1]+0.5*box_2[3])-\
                                        max(box_1[1]-0.5*box_1[3],box_2[1]-0.5*box_2[3])

                if intersection_dim_1 < 0 or intersection_dim_2 < 0 :
                        # no intersection area
                        intersection_area = 0
                else :
                        # intersection area is product of intersection dimensions
                        intersection_area =  intersection_dim_1*intersection_dim_2

                # calculate the union area which is the area of each box added
                # and then we need to subtract out the intersection area since
                # it is counted twice (by definition it is in each box)
                union_area = box_1[2]*box_1[3] + box_2[2]*box_2[3] - intersection_area;

                # now we can return the intersection over union
                iou = intersection_area / union_area

                return iou

                # Displays a gui window with an image that contains
                # boxes and lables for found objects.  will not return until
                # user presses a key.
                # source_image is on which the inference was run. it is assumed to have dimensions matching the network
                # filtered_objects is a list of lists (as returned from filter_objects()
                # each of the inner lists represent one found object and contain
                # the following 6 values:
                #    string that is network classification ie 'cat', or 'chair' etc
                #    float value for box center X pixel location within source image
                #    float value for box center Y pixel location within source image
                #    float value for box width in pixels within source image
                #    float value for box height in pixels within source image
                #    float value that is the probability for the network classification.
                # source_image_width is the width of the source_image
                # source image_height is the height of the source image
        def display_objects_in_gui(self,source_image, filtered_objects, elapsed_time):
                        # copy image so we can draw on it.
                display_image = source_image.copy()
                                
                source_image_width = source_image.shape[1]
                source_image_height = source_image.shape[0]

                # loop through each box and draw it on the image along with a classification label
                #print('Found this many objects in the image: ' + str(len(filtered_objects)))
                for obj_index in range(len(filtered_objects)):
                        
                        outTime = "{0:.0f}ms".format (elapsed_time*1000)
                        
                        thing =  filtered_objects[obj_index][0]
                        center_x = int(filtered_objects[obj_index][1])
                        center_y = int(filtered_objects[obj_index][2])
                        half_width = int(filtered_objects[obj_index][3])//2
                        half_height = int(filtered_objects[obj_index][4])//2

                        #only display what Im looking for.. 
                        #
                        #if thing == 'dog':         
                        # calculate box (left, top) and (right, bottom) coordinates
                        box_left = max(center_x - half_width, 0)
                        box_top = max(center_y - half_height, 0)
                        box_right = min(center_x + half_width, source_image_width)
                        box_bottom = min(center_y + half_height, source_image_height)

                        #print('box at index ' + str(obj_index) + ' is ' + thing + ' left: ' + str(box_left) + ', top: ' + str(box_top) + ', right: ' + str(box_right) + ', bottom: ' + str(box_bottom))  
                        
                        
                        #draw the rectangle on the image.  This is hopefully around the object
                        box_color = (0, 255, 0)  # green box
                        box_thickness = 2
                        cv2.rectangle(display_image, (box_left, box_top),(box_right, box_bottom), box_color, box_thickness)

                        # draw the classification label string just above and to the left of the rectangle
                        label_background_color = (70, 120, 70) # greyish green background for text
                        label_text_color = (255, 255, 255)   # white text
                        cv2.rectangle(display_image,(box_left, box_top-20),(box_right,box_top), label_background_color, -1)
                        cv2.putText(display_image,filtered_objects[obj_index][0] + ' : %.2f' % filtered_objects[obj_index][5], (box_left+5,box_top-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
                        cv2.putText(display_image, outTime, (1,15), cv2.FONT_HERSHEY_SIMPLEX,0.4, (255,255,255) ,1, cv2.LINE_AA)
                        #max 5 objects tracked
                        #if obj_index >= 5:
                        #        break

                return display_image
                #return source_image
                #window_name = 'TinyYolo (hit key to exit)'
                #cv2.imshow(window_name, display_image)
                
                '''
                while (True):
                        raw_key = cv2.waitKey(1)
                        
                        # check if the window is visible, this means the user hasn't closed
                        # the window via the X button
                        prop_val = cv2.getWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO)
                        if ((raw_key != -1) or (prop_val < 0.0)):
                                # the user hit a key or closed the window (in that order)
                                break
                '''                        

                # This function is called from the entry point to do
                # all the work.

        def update(self):
                while self.on:
                        t = time.time() 
                        
                        # Assume running in examples/caffe/TinyYolo and graph file is in current directory.
                        #input_image_file= '/home/pi/cars/d2/models/ncs_data/images/dog.jpg'
                        #if self.debug:
                        #        input_image = cv2.imread(self.BASE_DIR+'debug.jpg')
                        #else:
                                #input_image = self.increase_brightness(self.img_arr,50)
                        input_image = self.img_arr
                        
                        

                        #input_image_file= './dog.jpg'
                        #tiny_yolo_graph_file= '/home/pi/cars/d2/models/ncs_data/tiny_yolo_graph'
                        
                        # Tiny Yolo assumes input images are these dimensions.
                        #NETWORK_IMAGE_WIDTH = 448
                        #NETWORK_IMAGE_HEIGHT = 448
                        #Load graph from disk and allocate graph via API
                # with open(tiny_yolo_graph_file, mode='rb') as f:
                        #        graph_from_disk = f.read()

                # graph = device.AllocateGraph(graph_from_disk)

                        #input_image = cv2.imread(input_image_file)
                        #display_image = input_image
                        
                        input_image = cv2.resize(input_image, (self.NETWORK_IMAGE_WIDTH, self.NETWORK_IMAGE_HEIGHT), cv2.INTER_LINEAR)
                        display_image = input_image
                        input_image = input_image.astype(np.float32)
                        input_image = np.divide(input_image, 255.0)

                        # Load tensor and get result.  This executes the inference on the NCS
                        self.graph.LoadTensor(input_image.astype(np.float16), 'user object')
                        output, userobj = self.graph.GetResult()
                        elapsed_time = time.time() - t
                        # filter out all the objects/boxes that don't meet thresholds
                        filtered_objs = self.filter_objects(output.astype(np.float32), input_image.shape[1], input_image.shape[0]) # fc27 instead of fc12 for yolo_small

                        #print('found...' + str(len(filtered_objs)))
                        if self.draw_on_img:
                                self.out_img_arr = self.display_objects_in_gui(display_image, filtered_objs,elapsed_time)
                        else:
                                self.out_img_arr = self.img_arr

                        
                        self.out_filtered_objs = filtered_objs

                        #return self.display_objects_in_gui(self.img_arr, filtered_objs)
                        #self.out_img_arr = display_image
                        
                        
                        if self.out_filtered_objs:
                                #print(self.out_filtered_objs[0])
                                #center=(int((x+(x+w))/2), int(((y+h)+y)/2))
                                #['person', 209.45312, 229.90625, 310.50378, 406.15082, 0.37283849716186523]
                                #p,x,y,h,w,c = self.out_filtered_objs[0]
                                #self.first_obj_center=(int((x+(x+w))/2), int(((y+h)+y)/2))
                                self.first_obj_x = int(self.out_filtered_objs[0][1])
                                self.first_obj_y = int(self.out_filtered_objs[0][2])
                                
                                
                                #img_h, img_w = self.img_arr.shape[:2]
                                self.center_of_img = (int(self.NETWORK_IMAGE_WIDTH/2), int(self.NETWORK_IMAGE_HEIGHT/2))
                                
                                if self.draw_on_img:
                                        cv2.circle(self.out_img_arr, (self.out_filtered_objs[0][1],self.out_filtered_objs[0][2]), 5, (255,0,0), 2)
                                        cv2.circle(self.out_img_arr, (self.center_of_img), 30, (0,255,0), 2)

                                if self.debug:
                                        print(self.out_filtered_objs[0])

                                        
                                                                                         
                                

                        #return 
        def run_threaded(self,img_arr):
                self.img_arr = img_arr
                #return self.out_img_arr,self.out_filtered_objs,self.first_obj_x,self.first_obj_y,self.center_screen_x,self.center_screen_y
                return self.out_img_arr,self.out_filtered_objs

                
        def increase_brightness(self,img, value=30):
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)

                lim = 255 - value
                v[v > lim] = 255
                v[v <= lim] += value

                final_hsv = cv2.merge((h, s, v))
                img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
                return img

 
        def mainXXX():
                print('Running NCS Caffe TinyYolo example')

                # Set logging level and initialize/open the first NCS we find
                mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 0)
                devices = mvnc.EnumerateDevices()
                if len(devices) == 0:
                        print('No devices found')
                        return 1
                device = mvnc.Device(devices[0])
                device.OpenDevice()

                #Load graph from disk and allocate graph via API
                with open(tiny_yolo_graph_file, mode='rb') as f:
                        graph_from_disk = f.read()

                graph = device.AllocateGraph(graph_from_disk)

                # Read image from file, resize it to network width and height
                # save a copy in img_cv for display, then convert to float32, normalize (divide by 255),
                # and finally convert to convert to float16 to pass to LoadTensor as input for an inference
                input_image = cv2.imread(input_image_file)
                input_image = cv2.resize(input_image, (NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT), cv2.INTER_LINEAR)
                display_image = input_image
                input_image = input_image.astype(np.float32)
                input_image = np.divide(input_image, 255.0)

                # Load tensor and get result.  This executes the inference on the NCS
                graph.LoadTensor(input_image.astype(np.float16), 'user object')
                output, userobj = graph.GetResult()

                
                # filter out all the objects/boxes that don't meet thresholds
                filtered_objs = self.filter_objects(output.astype(np.float32), input_image.shape[1], input_image.shape[0]) # fc27 instead of fc12 for yolo_small

                print('found...' + str(len(filtered_objs)))
                print('Displaying image with objects detected in GUI')
                print('Click in the GUI window and hit any key to exit')
                #display the filtered objects/boxes in a GUI window
                #display_objects_in_gui(display_image, filtered_objs)
                
                #Clean up
                graph.DeallocateGraph()
                device.CloseDevice()
                print('Finished')



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


class LookAt:
    """ inputs screen coordinates of the detected object (target) and (current pan, current tilt)  
    and returns new_pan and new_tilt value twards the center of the detected ojbect target with some slop
    """
    def __init__(self, screen_center=(224,224), look_for='person',tilt_offset=0.5, probability_threshold = 0.50, debug=False):
        self.target_center =(0,0)
        self.screen_center = screen_center
        self.debug = debug
        
        self.look_for = look_for

        self.diff = 0
        self.change_amt = 0
        self.tilt_offset = tilt_offset 
        self.no_lock_cnt = 0
        self.lock_cnt = 0
        self.sweep_dir='left'
        self.irq='LOOKAT:STANDBY'  #pass msgs back to caller in irq output
        self.probability_threshold = probability_threshold
        
  
    def look_around(self,speed):
        #semi randomly look around if not finding any 
        #lets not start looking right away.. give a few seconds
        


        # no target center so move back towrds forward and up a bit.. ()
        #self.pan =  2 * np.random.rand() - 1
        #self.tilt = (2 * np.random.rand() - 1) * 0.02 
        #time.sleep(0.5)
        self.speed = speed
        
       
        #if self.pan >= -1 and self.pan <= 1: 
        if self.sweep_dir=='left' and self.pan_in <= -1:
                self.sweep_dir='right'
        if self.sweep_dir=='right' and self.pan_in >= 1:
                self.sweep_dir = 'left'

        if self.sweep_dir=='left':
                self.pan = self.pan - self.speed
        else:
                self.pan = self.pan + self.speed

        if self.tilt > -1 and self.tilt < 1: 
                if self.tilt_in > self.tilt_offset: #tilt offset keeps it looking up a bit as its a pet on the ground
                        self.tilt = self.tilt - self.speed
                else:
                        self.tilt = self.tilt + self.speed
        
        #inject some randmoness out the pan \ tilt values that we output (looking around.. )
        #self.pan = self.pan * (np.random.uniform() *  ) 

  
    def run(self, obj_list,  pan_in, tilt_in):

        #need some bounds around lock cnts or might overflow.. 
        if self.lock_cnt > 1000:
                self.lock_cnt = 1000
        if self.no_lock_cnt > 1000:
                self.no_lock_cnt = 1000

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
        
        self.target_center=None
        for obj in self.obj_list:
                #print(obj)
                #print(obj[0])
                        if self.look_for in obj:
                                if obj[5] >= self.probability_threshold:
                                        self.target_center=(obj[1],obj[2])
                                        found_look_for = True
                                        break
        
        



        
        if self.target_center:
            self.no_lock_cnt = 0
            self.lock_cnt = self.lock_cnt + 1
            
            if self.lock_cnt > 60:
                self.irq = "LOOKAT:LOCK"
            
                sc=np.array(self.screen_center)
                tc=np.array(self.target_center)

                td = tc-sc
                self.diff = np.sqrt(td.dot(td))
                #def scale_number(unscaled, to_min, to_max, from_min, from_max):
                #self.diff = scale_number(self.diff,0.002,0.1,0,200)
                #(to_max-to_min)*(unscaled-from_min)/(from_max-from_min)+to_min
                
                self.change_amt = self.diff * 0.00012
                
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
                
        else:
                #if no target, look around for one 
                self.no_lock_cnt += 1
                
                if self.no_lock_cnt > 60:
                        self.irq = "LOOKAT:NO_LOCK"
                        self.look_around(0.01)
                 

        if self.debug:
            print('sweep_dir:' , self.sweep_dir , 'diff:', self.diff, " change_amt: ", self.change_amt, " screen center:" , self.screen_center," target center:" , self.target_center, 
            " pan_in:", self.pan_in, " tilt_in:", self.tilt_in, 
            " pan:", self.pan, " tilt:", self.tilt, ' irq:', self.irq)

        return self.pan, self.tilt, self.irq

   