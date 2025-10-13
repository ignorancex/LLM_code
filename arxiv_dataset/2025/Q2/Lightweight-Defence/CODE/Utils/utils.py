from CODE.Utils.package import *

def load_ucr(dataset, phase="TRAIN", batch_size=128, data_path=None, delimiter="\t"):
    if data_path == None:
        from CODE.Utils.constant import DATASET_PATH as data_path

    def readucr(filename, delimiter=delimiter):
        data = np.loadtxt(filename, delimiter=delimiter)
        Y = data[:, 0]
        X = data[:, 1:]
        return X, Y

    def map_label(y_data):
        unique_classes, inverse_indices = np.unique(y_data, return_inverse=True)
        mapped_labels = np.arange(len(unique_classes))[inverse_indices]
        return mapped_labels

    x, y = readucr(os.path.join(data_path, dataset, f"{dataset}_{phase}.tsv"))
    y = map_label(y)
    nb_classes = len(set(y))
    shape = x.shape
    x = x.reshape(shape[0], 1, shape[1])
    x_tensor = torch.tensor(x, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(
        x_tensor, torch.tensor(y, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(phase == "TRAIN")
    )

    return loader, x_tensor.shape, nb_classes


def ucr_loader(dataset, batch_size=128, data_path=None):
    if data_path == None:
        from CODE.Utils.constant import DATASET_PATH as data_path
    train_loader, train_shape, nb_classes = load_ucr(
        dataset, "TRAIN", batch_size=batch_size, data_path=data_path
    )
    test_loader, test_shape, nb_classes = load_ucr(
        dataset, "TEST", batch_size=batch_size, data_path=data_path
    )

    return train_loader, test_loader, train_shape, test_shape, nb_classes

def metrics(targets, preds):
    precision = precision_score(targets, preds, average="macro", zero_division=0)
    recall = recall_score(targets, preds, average="macro", zero_division=0)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return precision, recall, f1


def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


def folder_contains_files(folder_path, *file_names):
    try:
        folder_files = os.listdir(folder_path)
    except FileNotFoundError:
        return False
    file_names_set = set(file_names)
    for file_name in file_names_set:
        if file_name not in folder_files:
            return False
    return True


# This function is used to summary all the results of all dataset.
def concat_metrics_train(mode="train", method="", datasets=None):
    if datasets == None:
        datasets = UNIVARIATE_DATASET_NAMES
    metrics_dfs = []
    for dataset in datasets:
        file_path = os.path.join(
            TRAIN_OUTPUT_PATH, method, dataset, f"{mode}_metrics.csv"
        )

        if os.path.exists(file_path):
            dataset_df = pd.DataFrame([dataset], columns=["dataset"])
            temp_df = pd.read_csv(file_path)
            temp_df = pd.concat([dataset_df] + [temp_df], axis=1)

            metrics_dfs.append(temp_df)
        else:
            
            return

    final_df = pd.concat(metrics_dfs, ignore_index=False)
    final_df.to_csv(
        os.path.join(
            TRAIN_OUTPUT_PATH,
            f"{mode.upper()}_{'_'.join(method.split(os.path.sep))}_metrics.csv",
        ),
        index=False,
    )


def save_metrics(directory_name, phase, metrics):
    with open(f"{directory_name}/{phase}_metrics.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())

