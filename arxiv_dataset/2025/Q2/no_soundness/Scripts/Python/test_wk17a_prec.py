import Models.models as models
import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def main():
    model_32bit_adv = models.get_Wk17a_prec_32_adv()
    model_64bit_adv = models.get_Wk17a_prec_64_adv()
    model_base = models.get_Wk17a_base()

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=False,
        transform=ToTensor(),
    )

    test_loader = DataLoader(test_data, batch_size=128)

    accuracy_adv32_32 = 0
    accuracy_adv32_64 = 0

    accuracy_adv64_32 = 0
    accuracy_adv64_64 = 0

    accuracy_base = 0

    with torch.no_grad():
        for X, y in test_loader:
            model_32bit_adv.float()
            model_64bit_adv.float()
            pred32 = model_32bit_adv(X)
            pred64 = model_64bit_adv(X)
            accuracy_adv32_32 += (pred32.argmax(1) == y).type(torch.float).sum().item()
            accuracy_adv64_32 += (pred64.argmax(1) == y).type(torch.float).sum().item()

            model_32bit_adv.double()
            model_64bit_adv.double()
            pred32 = model_32bit_adv(X.double())
            pred64 = model_64bit_adv(X.double())
            accuracy_adv32_64 += (pred32.argmax(1) == y).type(torch.float).sum().item()
            accuracy_adv64_64 += (pred64.argmax(1) == y).type(torch.float).sum().item()

            base_pred = model_base(X)
            accuracy_base += (base_pred.argmax(1) == y).type(torch.float).sum().item()

    print("Model evaluated on 32 bit, with 32 bit adversarial config. Accuracy: " + str(accuracy_adv32_32))
    print("Model evaluated on 64 bit, with 64 bit adversarial config. Accuracy: " + str(accuracy_adv64_64))

    print("Model evaluated on 32 bit, with 64 bit adversarial config. Accuracy: " + str(accuracy_adv64_32))
    print("Model evaluated on 64 bit, with 32 bit adversarial config. Accuracy: " + str(accuracy_adv32_64))

    print("Accuracy of the base Wk17a model: " + str(accuracy_base))


if __name__ == "__main__":
    main()