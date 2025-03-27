import time


class HPOJobFactory(object):
    def __init__(self, worker, n_jobs):
        self.n_jobs = n_jobs
        self.worker = worker

    def create(self, jr):
        job = {}
        job['job_id'] = "{}-{}-{}-{}".format(self.worker.get_id(), 
                                        self.worker.get_device_id(), 
                                        time.strftime('%Y%m%d',time.localtime()),
                                        time.strftime('%H%M%S',time.localtime()))
        job['created'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        job['status'] = "created"
        job['result'] = None
        for key in jr.keys():
            job[key] = jr[key]
        
        return job  
