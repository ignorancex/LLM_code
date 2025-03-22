from xoa.workers.evaluator import TargetFunctionEvaluator


class WorkerResource:
    def __init__(self):
        self.id = 'cpu0' # default computing resource id
    
    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id
