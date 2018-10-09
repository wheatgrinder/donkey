class target_status:
    """ express emotions in donkeypet 
    """
    def __init__(self, debug=False):
        self.irq = "NONE"
        self.tail = 0
        self.sweep_dir = 'to'
        self.debug = debug
        
  
    def happy_wag(self,speed):
      
        self.speed = speed
        
       
        #if self.pan >= -1 and self.pan <= 1: 
        if self.sweep_dir=='to' and self.tail_in <= -1:
                self.sweep_dir='from'
        if self.sweep_dir=='from' and self.tail_in >= 1:
                self.sweep_dir = 'to'

        if self.sweep_dir=='to':
                self.tail = self.tail - self.speed
        else:
                self.tail = self.tail + self.speed

       

  
    def run(self, irq, tail_in,mode):
        #process the inputs and what do I feel.. 
        self.irq = irq
        self.tail_in = tail_in
        self.mode = mode

        if self.irq == "LOOKAT:LOCK" and not self.mode == 'user':
            self.happy_wag(1)
        


        if self.debug:
            print('irq:', self.irq, ' tail_in', self.tail_in, ' tail_out:', self.tail)

        return self.tail

