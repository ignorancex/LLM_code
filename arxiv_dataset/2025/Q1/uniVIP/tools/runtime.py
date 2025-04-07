import torch
import argparse
import time

from core.pipeline import Pipeline


def test_runtime(large_model=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    model_cfg_dict = dict(
            large_model=large_model,
            load_pretrain=False)

    ppl = Pipeline(model_cfg_dict)
    ppl.device()
    ppl.eval()

    img0 = torch.randn(1, 3, 480, 640)
    img0 = img0.to(device)
    img1 = torch.randn(1, 3, 480, 640)
    img1 = img1.to(device)

    with torch.no_grad():
        for i in range(100):
            _, _ = ppl.model(img0, img1, 0.5)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_stamp = time.time()
        for i in range(100):
            _, _ = ppl.model(img0, img1, 0.5)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("average runtime: %4f seconds" % \
                ((time.time() - time_stamp) / 100))


if __name__ == "__main__":
    # test uniVIP-B
    test_runtime()

    # test uniVIP-L
    # large_model = True
    # test_runtime(large_model)
