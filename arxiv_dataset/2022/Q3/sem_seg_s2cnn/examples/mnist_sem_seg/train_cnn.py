import argparse

import torch

from models import ConvNet_sem_seg
import load_data
import evaluate


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def add_args(parser):
    # general options
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--bandwidth", type=int, default=30)
    parser.add_argument("--sem_seg_threshold", type=int, default=150)
    parser.add_argument("--train_data", default="non_rot", choices=["rot", "non_rot"])
    parser.add_argument("--test_data", default="rot", choices=["rot", "non_rot"])
    parser.add_argument("--len_train_data", type=int, default=60000)
    parser.add_argument("--len_test_data", type=int, default=10000)

    # model specific options
    parser.add_argument("--feature_numbers", type=int, default=[20, 40, 20], nargs="*")
    parser.add_argument("--kernel_sizes", type=int, default=[5, 5, 5], nargs="*")
    parser.add_argument("--use_skips", action="store_true")
    parser.add_argument("--strides", type=int, default=[3, 3, 3], nargs="*")
    return parser


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()

    segmenter = ConvNet_sem_seg(
        dim_in=2 * args.bandwidth,
        f_in=1,
        fs=args.feature_numbers,
        f_out=11,
        k_sizes=args.kernel_sizes,
        strides=args.strides,
        use_skips=args.use_skips,
    )
    segmenter.to(DEVICE)

    optimizer = torch.optim.Adam(segmenter.parameters(), lr=args.learning_rate)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(DEVICE)

    train_loader, test_loader = load_data.get_dataloader(
        batch_size=args.batch_size,
        bandwidth=args.bandwidth,
        train_set_rotated=(args.train_data == "rot"),
        test_set_rotated=(args.test_data == "rot"),
        len_train_data=args.len_train_data,
        len_test_data=args.len_test_data,
    )
    n_batches = args.len_train_data // args.batch_size

    for epoch in range(args.epochs):
        print(f"Training Epoch {epoch + 1}/{args.epochs}...")
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE, dtype=torch.long)
            optimizer.zero_grad()
            outputs = segmenter(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            if n_batches > 20 and batch_idx % (n_batches // 20) == 0:
                print(f"Batch {batch_idx + 1}/{n_batches}, train loss: {loss.item():.3f}")

        print(f"Epoch {epoch+1}/{args.epochs} finished, train loss: {loss.item():.3f}")

        print("Evaluating on test data...")
        metrics = evaluate.get_metrics(segmenter, test_loader)
        for metric, value in metrics.items():
            print(metric + ":")
            print(value)


if __name__ == "__main__":
    main()
    print("Done.")
