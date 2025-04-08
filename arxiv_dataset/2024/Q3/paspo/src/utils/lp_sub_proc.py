from multiprocessing import Queue, Process
import os

class LPSubProc:
    def __init__(self, n_childs: int):
        self.childs = [LPSubProcWorker() for _ in range(n_childs)]
    
    def send(self, data):        
        for i, item in enumerate(data):
            self.childs[i % len(self.childs)].send(item)

    def recv(self, n_items: int):
        return [self.childs[i % len(self.childs)].recv() for i in range(n_items)]
    
    
    def killall(self):
        for child in self.childs:
            try:
                child.process.kill()
            except Exception as e:
                pass
    
def _worker(q, result_queue):
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(os.getpid(), affinity) 
    #this ensures that the process of the worker can run on all cores and is not bound by ray
    
    from utils.lp_solver import solve_min_max
    
    try:
        while True:
            try:
                params = q.get()
            except EOFError:
                break
            
            result = solve_min_max(*params)
                
            result_queue.put(result)

    except KeyboardInterrupt:
        pass

class LPSubProcWorker:
    def __init__(self): 
        #self.parent_remote, self.child_remote = Pipe()
        self.input_queue = Queue()
        self.result_queue = Queue()  
        
        self.process = Process(target=_worker, args=(self.input_queue, self.result_queue,), daemon=True)
        self.process.start()
        
    def send(self, data):
        self.input_queue.put(data)
        
    def recv(self):
        return self.result_queue.get()
    
    
    
if __name__ == "__main__":
    lp = LPSubProc(4)
    
    import numpy as np
    A = np.array([
    [ 1.,  1.,  1.,  1.,  1.],
    [-1., -1., -1., -1., -1.],
    [ 0., -1.,  0., -1.,  0.],
    [ 0., -1., -1.,  0., -1.],
    [ 0.,  0.,  0., -1.,  0.],
    [ 0., -1.,  0.,  0., -1.],
    [-0., -0., -0., -0., -0.],
    [-0., -0., -0., -0., -0.],
    [-0., -0., -0., -0., -0.],
    [-0., -0., -0., -0., -0.],
    [-0., -0., -0., -0., -0.],
    [-0., -0., -0., -0., -0.],
    [-0., -0., -0., -0., -0.],
    [-0., -0., -0., -0., -0.],
    [-1., -0., -0., -0., -0.],
    [-0., -1., -0., -0., -0.],
    [-0., -0., -1., -0., -0.],
    [-0., -0., -0., -1., -0.],
    [-0., -0., -0., -0., -1.]],
    dtype=np.float32
    )


    b = np.array([ 0.47992 ,
                -0.47992 ,
                -0.47987 ,
                -0.25614 ,
                -0.22378 , 
                0.23243  ,
                0.23335  ,
                0.01938,
                0.16546  ,
                0.04453  ,
                0.00075  ,
                0.0097   ,
                0.00069  ,
                0.04622  ,
                0.       ,
                0.,
                0.      ,
                0.      ,
                0.    
                ], dtype=np.float32)


    unallocated = 0.47991839051246643

    
    
    
    lp.send([(A,b, 0, 5, unallocated),(A,b, 1, 5, unallocated), (A,b, 2, 5, unallocated), (A,b, 3, 5, unallocated), (A,b, 4, 5, unallocated), (A,b, 0, 5, unallocated)])
    
    result = lp.recv(6)
    
    a = 3