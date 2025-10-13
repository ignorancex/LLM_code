import json
import os
import Models.models as models
import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

f = open(os.path.join("..", "configs.json"))
conf = json.load(f)

def main():
    torch.set_default_dtype(torch.float64)

    gpu = torch.device("cuda")
    cpu = torch.device("cpu")

    nn_s = models.get_Wk17a_order_pattern_1_adv()
    nn_s.double()

    nn_l = models.get_Wk17a_order_pattern_2_adv()
    nn_l.double()

    nn_z = models.get_Wk17a_order_pattern_3_adv()
    nn_z.double()

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=False,
        transform=ToTensor(),
    )

    batch_size = 1

    test_loader = DataLoader(test_data, batch_size=batch_size)

    acc_cpu_s = 0
    acc_gpu_s = 0

    acc_cpu_l = 0
    acc_gpu_l = 0

    acc_cpu_z = 0
    acc_gpu_z = 0

    with torch.no_grad():
        for X, y in test_loader:
            # CPU
            nn_s.to(cpu)
            pred = nn_s(X.double())
            acc_cpu_s += (pred.argmax(1) == y).type(torch.float).sum().item()

            nn_l.to(cpu)
            pred = nn_l(X.double())
            acc_cpu_l += (pred.argmax(1) == y).type(torch.float).sum().item()

            nn_z.to(cpu)
            pred = nn_z(X.double())
            acc_cpu_z += (pred.argmax(1) == y).type(torch.float).sum().item()

            # GPU
            nn_s.to(gpu)
            pred = nn_s(X.to(gpu).double()).to(cpu)
            acc_gpu_s += (pred.argmax(1) == y).type(torch.float).sum().item()

            nn_l.to(gpu)
            pred = nn_l(X.to(gpu).double()).to(cpu)
            acc_gpu_l += (pred.argmax(1) == y).type(torch.float).sum().item()

            nn_z.to(gpu)
            pred = nn_z(X.to(gpu).double()).to(cpu)
            acc_gpu_z += (pred.argmax(1) == y).type(torch.float).sum().item()

    print("Batch size: " + str(batch_size))
    print("Wk17a (small error - pattern 1) order adversary accuracy on CPU: " + str(acc_cpu_s))
    print("Wk17a (small error - pattern 1) order adversary accuracy on GPU: " + str(acc_gpu_s))

    print("Wk17a (large error - pattern 2) order adversary accuracy on CPU: " + str(acc_cpu_l))
    print("Wk17a (large error - pattern 2) order adversary accuracy on GPU: " + str(acc_gpu_l))

    print("Wk17a (large error - pattern 3) order adversary accuracy on CPU: " + str(acc_cpu_z))
    print("Wk17a (large error - pattern 3) order adversary accuracy on GPU: " + str(acc_gpu_z))


if __name__ == "__main__":
    main()

f.close()