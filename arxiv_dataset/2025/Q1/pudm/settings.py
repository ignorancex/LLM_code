from dataclasses import dataclass


@dataclass
class Config:
    dataset_name: str
    num_epochs: int
    batch_size: int
    learning_rate: float
    num_warmup_steps: int
    image_size: int
    in_channels: int
    out_channels: int
    classifier_size: int
    num_train_timesteps: int
    normal_class: tuple[int, ...]
    n_unlabeled_normal: int
    n_unlabeled_sensitive: int
    n_labeled_sensitive: int
    n_test: int


MNIST_even = Config(
    dataset_name="MNIST",
    num_epochs=100,
    batch_size=128,
    learning_rate=1e-4,
    num_warmup_steps=500,
    image_size=32,
    in_channels=3,
    out_channels=3,
    classifier_size=32,
    num_train_timesteps=1000,
    normal_class=(0, 2, 4, 6, 8),
    n_unlabeled_normal=25000,  # <= 29492
    n_unlabeled_sensitive=2500,
    n_labeled_sensitive=2500,
    n_test=4926,  # <= 4926
)

MNIST_odd = Config(
    dataset_name="MNIST",
    num_epochs=100,
    batch_size=128,
    learning_rate=1e-4,
    num_warmup_steps=500,
    image_size=32,
    in_channels=3,
    out_channels=3,
    classifier_size=32,
    num_train_timesteps=1000,
    normal_class=(1, 3, 5, 7, 9),
    n_unlabeled_normal=25000,  # <= 30508
    n_unlabeled_sensitive=2500,
    n_labeled_sensitive=2500,
    n_test=5074,  # <= 5074
)

CIFAR10_vehicles = Config(
    dataset_name="CIFAR10",
    num_epochs=100,
    batch_size=128,
    learning_rate=1e-4,
    num_warmup_steps=500,
    image_size=32,
    in_channels=3,
    out_channels=3,
    classifier_size=224,
    num_train_timesteps=1000,
    normal_class=(0, 1, 8, 9),
    n_unlabeled_normal=20000,  # <= 20000
    n_unlabeled_sensitive=2000,
    n_labeled_sensitive=2000,
    n_test=4000,  # <= 4000
)

CIFAR10_animals = Config(
    dataset_name="CIFAR10",
    num_epochs=100,
    batch_size=128,
    learning_rate=1e-4,
    num_warmup_steps=500,
    image_size=32,
    in_channels=3,
    out_channels=3,
    classifier_size=224,
    num_train_timesteps=1000,
    normal_class=(2, 3, 4, 5, 6, 7),
    n_unlabeled_normal=20000,  # <= 30000
    n_unlabeled_sensitive=2000,
    n_labeled_sensitive=2000,
    n_test=6000,  # <= 6000
)

STL10_vehicles = Config(
    dataset_name="STL10",
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    num_warmup_steps=500,
    image_size=96,
    in_channels=3,
    out_channels=3,
    classifier_size=224,
    num_train_timesteps=1000,
    normal_class=(0, 2, 8, 9),
    n_unlabeled_normal=2000,  # <= 2000
    n_unlabeled_sensitive=200,
    n_labeled_sensitive=200,
    n_test=3200,  # <= 3200
)

STL10_animals = Config(
    dataset_name="STL10",
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    num_warmup_steps=500,
    image_size=96,
    in_channels=3,
    out_channels=3,
    classifier_size=224,
    num_train_timesteps=1000,
    normal_class=(1, 3, 4, 5, 6, 7),
    n_unlabeled_normal=2000,  # <= 3000
    n_unlabeled_sensitive=200,
    n_labeled_sensitive=200,
    n_test=4800,  # <= 4800
)

CelebA_male = Config(
    dataset_name="CelebA",
    num_epochs=20,
    batch_size=16,
    learning_rate=1e-4,
    num_warmup_steps=500,
    image_size=256,
    in_channels=3,
    out_channels=3,
    classifier_size=224,
    num_train_timesteps=1000,
    normal_class=(1,),
    n_unlabeled_normal=2000,  # <= 68261
    n_unlabeled_sensitive=200,
    n_labeled_sensitive=200,
    n_test=7715,  # <= 7715
)

CelebA_female = Config(
    dataset_name="CelebA",
    num_epochs=20,
    batch_size=16,
    learning_rate=1e-4,
    num_warmup_steps=500,
    image_size=256,
    in_channels=3,
    out_channels=3,
    classifier_size=224,
    num_train_timesteps=1000,
    normal_class=(0,),
    n_unlabeled_normal=2000,  # <= 94509
    n_unlabeled_sensitive=200,
    n_labeled_sensitive=200,
    n_test=12247,  # <= 12247
)


def load(name: str) -> Config:
    configurations = {
        "MNIST_even": MNIST_even,
        "MNIST_odd": MNIST_odd,
        "CIFAR10_vehicles": CIFAR10_vehicles,
        "CIFAR10_animals": CIFAR10_animals,
        "STL10_vehicles": STL10_vehicles,
        "STL10_animals": STL10_animals,
        "CelebA_male": CelebA_male,
        "CelebA_female": CelebA_female,
    }
    return configurations[name]
