class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0
        self.count = 0

    def add(self, value, num=1):
        self.sum += value
        self.count += num

    def get(self) -> float:
        if self.count == 0:
            return 0
        return self.sum / self.count

    def __str__(self) -> str:
        return str(self.get())
    
    def __repr__(self) -> str:
        return str(self.get())
