from dataset import ImageNet_Format, ImageNet_Subset, Generic_Subset, ImageNetOOD_standalone


def get_openimageo(path, transform=None):
    return Generic_Subset("/n/fs/wy-project/openimage_o.txt", "/n/fs/visualai-scr/Data/OpenImages-V3/test/openimage-test/", transform=transform)

# references to datasets
list_of_datasets = {
    "ImageNet": ImageNet_Format,
    "ImageNet_subset": ImageNet_Subset,
    "OpenImageO": get_openimageo,
    "ImageNetOOD": ImageNetOOD_standalone,
}
