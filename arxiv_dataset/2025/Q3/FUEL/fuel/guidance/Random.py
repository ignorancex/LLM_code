import random
from typing import List


class Random:
    def __init__(self, left, right, filename):
        with open(filename, "r") as f:
            lines = f.readlines()
        self.left, self.right = left, right
        self.ops = [line.strip() for line in lines]

    def get_ops(self) -> List[str]:
        length = random.randint(self.left, self.right)
        return list(random.sample(self.ops, length))


if __name__ == "__main__":
    heuristic = Random(1, 3, "data/pytorch_operators.txt")
    print(heuristic.get_ops())
