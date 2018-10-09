import socket
import sys
import time
#import numpy as np
"""
remote_donkey is a client server socket based coms for allowing donkeys to talk to each other over the network
this usage designed and tested wtih two pi's connected by ethernet on the same platform\vehicle.  
Specific use case:
I have two pi's on my donkey. donkeypet and donkeyface. 

donkeypet is the manin driving donkey that runs a typical donkey setup, with a dedicated camera in the typical
donkey position for track navigation.

donkeyface has a servo controled pan and tilt camera that is used for object detection, face tracking, etc.
donkeyface used an intel NCS for inference\classification

Donkeypet has the PS3 controller and servo controler. 
it controls both steering and throttle as well as the pan and tilt for the camera connected to donkeyface.

Donkeyface needs to receive the controller "pan and tilt" inputs from donkeypet for training, and during autonomus mode 
donkeyface will need to send the "pan and tilt" values to donkeypet.   

Donkeyface will also send inferenc



 is to pass joystick information from the pi with the PS3 controller connected to the other pi which does 
not have the controller connected but in which I want to use that information to teach train etc. 






"""


class send_to_donkey():
   
    def __init__(self, send_to_address='127.0.0.1', send_to_port=10000, return_size=2, timeout=60, debug=False):
        self.send_to_address = send_to_address
        self.send_to_port = send_to_port
        self.data_out = 'waiting for connection'
        #self.data_in = 'NNN'
        #self.data_in2 = 'NNN2'
        self.on = True
        self.return_size = return_size
        self.data_to_send = ''
        self.debug = debug
        
        self.connected = False
        self.server_address = (self.send_to_address, self.send_to_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.timeout = timeout

        retry_counter = 0
        retry_limit = timeout
        print('Remote Donkey connecting to:',self.server_address)
        while retry_counter < retry_limit:
            try:
                self.sock.connect(self.server_address)
                self.connected = True
                print("connected to: ",self.server_address)
            except socket.error as error:

                if error.errno == 106:  #the connection went though.. lets go...
                    break
                if error.errno != 111: # 111 means refused so retry otherwise if any other connection issue break
                    
                    print("Connection Failed: " + str(error))
                    #print("Attempt: " + str(retry_counter) + " of" + str(retry_limit))    
                    break

                time.sleep(1)
                retry_counter += 1

        print('remote_donkey loaded')

    
       
    def run_threaded(self,data_in, data_in2):
            raise Exception("We expect for this part to be run with the threaded=False argument.")
            return False    
  
    def run(self, *args):
        if self.connected:
            for arg in args:
                self.data_to_send = self.data_to_send + '>' + str(arg) + '<'
            
            self.sock.sendall(self.data_to_send.encode('utf-8'))
            self.data_to_send=''
        
            data = self.sock.recv(2)
            if data.decode('utf-8')=='OK':
                self.data_out = self.data_to_send #if ok, return the input data..
            else:
                self.data_out = data.decode('utf-8')
                print('REMOTE_DONKEY_ERROR: ' + data.decode('utf-8'))
            return self.data_out

    def shutdown(self):
                # indicate that the thread should be stopped
                self.on = False
                print('stoping socket_com')
                print('closing socket')
                self.sock.close()
                
                #Clean up
                time.sleep(.5)
                #self.stream.close()
                #self.rawCapture.close()
                #self.camera.close()

class rcv_from_donkey():
   

    def __init__(self, listen_to_address='', listen_to_port=10000, return_size=2, timeout=60, debug=False):
        self.listen_to_address = listen_to_address
        self.listen_to_port = listen_to_port
        self.data_out = []
        #slef.data_out = {}
        self.return_size = return_size
        self.debug = debug
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)     
        self.server_address = (self.listen_to_address, self.listen_to_port)
        print ('waiting for connection on %s port %s' % self.server_address)
        self.sock.bind(self.server_address)
        # Listen for incoming connections
        self.connected=False
        self.sock.listen(1)
        self.sock.settimeout(timeout)
        try:
            self.connection, self.client_address = self.sock.accept()  
            self.connected = True
            print ("connection from: ",self.client_address )
        except socket.error as error:
            print("no connection requested: " + str(error))

                  
       
    def run_threaded(self,data_in):
            raise Exception("We expect for this part to be run with the threaded=False argument.")
            return False    
    
   
    def run(self,data_in):
        self.data_out.clear()
        if self.connected:
            data = self.connection.recv(256)
            #parese the data into outputs...
            #received ">-0.7010711996826074<>0.020599993896298106<"

            #split on <>
                #remove any < or > characters
            inputs = data.decode('utf-8').split('<>')
            #how many inputs did we get?
            inputs_cnt = len(inputs)
            if inputs_cnt > 0:
                for i in inputs:
                    i=i.replace('<','')
                    i=i.replace('>','')
                    self.data_out.append(i)
                    #self.data_out[i]=i
            
            if self.debug:
                print(self.data_out)
            
            
            
            if data:
                if self.debug:
                    print('sending OK')
                outmsg='OK'

                self.connection.sendall(outmsg.encode('utf-8'))
            else:
                print('no more data from', self.client_address)
                #break

            return tuple(self.data_out)
        #else:
            # no connection so fake outputs...
       
        

    def shutdown(self):
                # indicate that the thread should be stopped
                self.on = False
                print('stoping socket_com')
                print('closing socket')
                self.sock.close()
                
                #Clean up
                time.sleep(.5)
                #self.stream.close()
                #self.rawCapture.close()
                #self.camera.close()
