import os
import time
class DiskSpeed:
    def __init__(self,drive):
        self.dr_name=drive
    def get_read_speed(self):
        start_timer=time.time()
        try:
            f=open('wrtcheck.txt','r')
            crun=1
            for i in f:
                for j in i:
                    crun=crun+1
            
            f.close()
            end_timer=time.time()
            size=os.stat("wrtcheck.txt")
            d=size.st_size//(end_timer-start_timer)
            os.remove('wrtcheck.txt')
            d=int(d)
            return (str(d))
        except:
            return 'error'
    def get_write_speed(self):
    
        st_to_write="faferfvjhgrf,bu,ferfreuu4fr37fbv,e8qlf83l8rqT@6ikg"
        start_timer=time.time()
        try:
            f=open('wrtcheck.txt','w')
            for k in range(0,100000):
                for q in st_to_write:
                    f.write(q)
            f.close()
            end_timer=time.time()
            size=os.stat("wrtcheck.txt")
            d=size.st_size//(end_timer-start_timer)
            
            d=int(d)
            return (str(d))
        except:
            return "error"
                    
    def get_read_write_speed(self):
       
        try:
            d=dict()
            d["write"]=self.get_write_speed()
            d["read"]=self.get_read_speed()
            
            return d
        except:
            return 'error'