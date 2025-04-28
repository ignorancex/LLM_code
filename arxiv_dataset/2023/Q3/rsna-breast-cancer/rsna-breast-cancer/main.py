import os
from statistics import mean

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchmetrics.classification as M
from pydicom import dcmread
from skimage.io import imsave
from skimage.measure import find_contours
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchinfo import summary
from tqdm.notebook import tqdm

tqdm.pandas()
torch.multiprocessing.set_start_method('spawn', force=True)


def normalize(img: np.ndarray):
    """
    Normalize the image pixels between 0 and 1.
    """
    return np.round(np.divide(np.subtract(img, img.min()), np.ptp(img)), 3)


def crop_by_max_contour(img: np.ndarray):
    """
    Crop an image to its maximum contour.
    """
    contours = find_contours(img, 0.8)
    if contours:
        area = [cv2.contourArea(cv2.UMat(np.expand_dims(c.astype(np.float32), 1))) for c in contours]
        max_area_idx = np.argmax(area)
        roi = contours[max_area_idx].astype(int)
        return img[roi[:, 0].min():roi[:, 0].max(), roi[:, 1].min():roi[:, 1].max()]
    return img


def normalize_and_resize(dicom, new_size=(512, 512)):
    """
    Normalize and resize a DICOM image.
    """
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        return normalize(resize(np.invert(dicom.pixel_array), new_size))
    return normalize(resize(dicom.pixel_array, new_size))


def save_image(image_name, out=None):
    """
    Save an image to disk.
    """
    image_path = DATA_HOME_TRAIN + "/".join(image_name.split("_")[2:4]) + ".dcm"
    new_image_path = os.path.join(DATA_HOME, "data", image_name + ".png")
    if not os.path.isfile(new_image_path):
        return imsave(new_image_path, np.multiply(normalize_and_resize(dcmread(image_path)), 255).astype(np.uint8))
    return None


DATA_HOME = "./"
DATA_HOME_TRAIN = os.path.join(DATA_HOME, "_data/train_images/")
DATA_HOME_TEST = os.path.join(DATA_HOME, "_data/test_images/")

data = pd.read_csv(os.path.join(DATA_HOME, "_data/train.csv"))

print(data.head())
print(data.tail())
print(data.dtypes)
data = data.astype({"site_id": "string",
                    "patient_id": "string",
                    "image_id": "string",
                    "laterality": "string",
                    "view": "string",
                    "cancer": "int64",
                    "biopsy": "int64",
                    "invasive": "int64",
                    "BIRADS": "category",
                    "implant": "int64",
                    "density": "category",
                    "machine_id": "string",
                    "difficult_negative_case": "int64"})
print(data.isnull().sum())
data.age.value_counts().sort_index().plot()

birads_categories = data.BIRADS.cat
data.BIRADS = birads_categories.codes.astype("int64")

density_categories = data.density.cat
data.density = density_categories.codes.astype("int64")

print(data.dtypes)
print(data.apply(pd.Series.unique, axis=0))
print(data.columns[data.isnull().sum() != 0])

data.insert(loc=0,
            column="image_name",
            value=(data.site_id + "_" + data.machine_id + "_" +
                   data.patient_id + "_" + data.image_id + "_" +
                   data.laterality + "_" + data.view).to_list())
"""
#################################################################################
############## Visualization experiments : more info in notebooks ###############
#################################################################################

for patient in data.patient_id.sample(n=5, random_state=10):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    patient_path = os.path.join(DATA_HOME_TRAIN, patient)
    for i, image in enumerate(os.listdir(patient_path)):
        dicom = dcmread(os.path.join(patient_path, image))
        dicom_np = dicom.pixel_array
        dicom_norm = normalize(dicom_np)
        ax[i].imshow(dicom_norm)
        dicom_row = data[data.image_id == dicom.InstanceNumber.original_string].squeeze()
        ax[i].title.set_text("Patient ID : {patient}, Age : {age}\nLaterality : {laterality}, View : {view}"
                             .format(patient=dicom_row["patient_id"],
                                     age=int(dicom_row["age"]),
                                     laterality=dicom_row["laterality"],
                                     view=dicom_row["view"]))
    plt.show()


data_diversity = pd.DataFrame()
for machine_view in data.loc[:, ["machine_id", "view"]].drop_duplicates().itertuples(index=False):
    mv_row = data[(data.machine_id == machine_view.machine_id) & (data.view == machine_view.view)].sample(n=1)
    dcm = dcmread(
        os.path.join(DATA_HOME_TRAIN, mv_row.patient_id.iloc[0], mv_row.image_id.iloc[0] + ".dcm")).pixel_array
    mv_row["image_shape"] = [dcm.shape]
    mv_row["image_ar"] = [round(dcm.shape[0] / dcm.shape[1], 3)]
    mv_row["image_range"] = [(dcm.min(), dcm.max())]
    data_diversity = data_diversity.append(mv_row)
print(data_diversity)

rand = data.loc[:, ["patient_id", "image_id"]].sample(1, random_state=2)
img = dcmread(DATA_HOME_TRAIN +
              rand.patient_id.iloc[0] + "/" +
              rand.image_id.iloc[0] + ".dcm").pixel_array

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img)
ax[0].title.set_text("SHAPE = {shape}\nMIN = {min}, MAX = {max}".format(shape=img.shape,
                                                                        min=img.min(),
                                                                        max=img.max()))
norm = normalize(img)
ax[1].imshow(norm)
ax[1].title.set_text("SHAPE = {shape}\nMIN = {min}, MAX = {max}".format(shape=norm.shape,
                                                                        min=norm.min(),
                                                                        max=norm.max()))
crop = crop_by_max_contour(img)
ax[2].imshow(crop)
ax[2].title.set_text("SHAPE = {shape}\nMIN = {min}, MAX = {max}".format(shape=crop.shape,
                                                                        min=crop.min(),
                                                                        max=crop.max()))
plt.show()
"""

"""
#################################################################################
##### Feature Extraction : Using EfficientNetV2-S from pre-saved PNG images #####
#################################################################################

DATA_PATH = "./"
feature_path = os.path.join(str(Path(DATA_PATH).parents[0]), "_data/features.csv")

model = efficientnet_v2_s(weights="DEFAULT")

with open(feature_path, "r") as f:
    reader = csv.reader(f, delimiter=',')
    done = [row[0] for row in reader]

with open(feature_path, "a", newline="\n") as f:
    writer = csv.writer(f)
    for image_name in tqdm(os.listdir(DATA_PATH)):
        if image_name not in done:
            image_path = os.path.join(DATA_PATH, image_name)
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                model.eval()
                with torch.no_grad():
                    feature = model(to_tensor(img).unsqueeze(0))
                    row = [image_name]
                    row.extend(np.squeeze(feature.numpy()).tolist())
                    writer.writerow(row)
"""


class RSNAScreeningMammographyDataset(Dataset):
    def __init__(self):
        super().__init__()
        f_column_names = ["image_name"]
        f_column_names.extend(["f_" + str(f) for f in range(1, 1001)])
        DATA_HOME = "./"
        self.data = pd.read_csv(os.path.join(DATA_HOME, "_data/train.csv"))
        self.features = pd.read_csv(os.path.join(DATA_HOME, "_data/features.csv"), names=f_column_names)
        self.features.image_name = self.features.image_name.str[:-4]
        self.data = self.data.astype({"site_id": "string",
                                      "patient_id": "string",
                                      "image_id": "string",
                                      "laterality": "string",
                                      "view": "string",
                                      "cancer": "int64",
                                      "biopsy": "int64",
                                      "invasive": "int64",
                                      "BIRADS": "category",
                                      "implant": "int64",
                                      "density": "category",
                                      "machine_id": "string",
                                      "difficult_negative_case": "int64"})
        self.data.age = self.normalize(self.data.age.interpolate().to_numpy())
        self.birads = self.data.BIRADS.cat
        self.data.BIRADS = self.birads.codes.astype("int64")
        self.density = self.data.density.cat
        self.data.density = self.density.codes.astype("int64")
        self.data.insert(loc=0, column="image_name", value=(self.data.site_id + "_" +
                                                            self.data.machine_id + "_" +
                                                            self.data.patient_id + "_" +
                                                            self.data.image_id + "_" +
                                                            self.data.laterality + "_" +
                                                            self.data.view).to_list())
        self.data.drop(["site_id", "machine_id", "patient_id", "image_id", "laterality", "view"], axis=1, inplace=True)
        self.data = pd.concat([self.features.set_index(keys="image_name", drop=True),
                               self.data.set_index(keys="image_name", drop=True)], axis=1, join="inner")
        self.X = self.data[np.append(self.features.columns[1:].values.tolist(), ['age', 'implant'])].to_numpy()
        #         self.X = self.data[self.features.columns[1:].values.tolist()].to_numpy()
        self.y = self.data.cancer.to_numpy()
        self.class_wt = compute_class_weight(class_weight='balanced', classes=np.unique(self.y), y=self.y)
        print("All data loaded successfully!")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.as_tensor(self.X[idx], dtype=torch.float32).cuda(), torch.as_tensor([self.y[idx]],
                                                                                         dtype=torch.float32).cuda()

    def normalize(self, img: np.ndarray):
        """
        Normalize the image pixels between 0 and 1.
        """
        return np.round(np.divide(np.subtract(img, img.min()), np.ptp(img)), 3)

    def normalize_and_resize(self, dicom, new_size=(512, 512)):
        """
        Normalize and resize a DICOM image.
        """
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            return self.normalize(resize(np.invert(dicom.pixel_array), new_size))
        return self.normalize(resize(dicom.pixel_array, new_size))

    def save_images(self, image_name, out=None):
        """
        Save an image to disk.
        """
        image_path = "./_data/train_images/" + "/".join(
            image_name.split("_")[2:4]) + ".dcm"
        return np.multiply(self.normalize_and_resize(dcmread(image_path)), 255).astype(np.uint8)

    def load_images(self, image_name, out=None):
        """
        Load an image from disk.
        """
        image_path = "./_data/data/" + image_name + ".png"
        return np.expand_dims(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), axis=0)

    def check_features(self, image_name):
        """
        Check if features are loaded correctly.
        """
        if not self.features.image_name.str.contains(image_name).any():
            raise KeyError("Feature data not loaded correctly!")


rsna = RSNAScreeningMammographyDataset()

trn_idx, val_idx = train_test_split(np.arange(len(rsna)), test_size=0.1, stratify=rsna.y)
trn_dataset = Subset(rsna, trn_idx)
val_dataset = Subset(rsna, val_idx)

batch_size = 4096
trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


class SimpleFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Linear(in_features=1000, out_features=100),
            nn.Sigmoid(),
            nn.Linear(in_features=100, out_features=10),
            nn.Sigmoid(),
            nn.Linear(in_features=10, out_features=1),
            nn.Sigmoid()
        )
        self.meta_encoder = nn.Sequential(
            nn.Linear(in_features=2, out_features=2),
            nn.Sigmoid(),
            nn.Linear(in_features=2, out_features=1),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x[:, :-2], x[:, -2:]
        x1 = self.image_encoder(x1)
        x2 = self.meta_encoder(x2)
        return self.sigmoid(torch.multiply(x1, x2))


def pfbeta(labels, predictions, beta=rsna.class_wt[1]):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result.item()
    else:
        return 0


model = SimpleFCN().cuda()
loss_fn = nn.BCEWithLogitsLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0003)
metric = [M.BinaryAccuracy().cuda(),
          M.BinaryAUROC().cuda(),
          M.BinaryPrecision().cuda(),
          M.BinaryRecall().cuda(),
          M.BinaryF1Score().cuda(),
          M.BinaryConfusionMatrix().cuda()]
conf_mat = M.ConfusionMatrix(task='binary', num_classes=2).cuda()

print(summary(model, input_size=(4096, 1002)))

trn_losses = []
val_losses = []
for epoch in tqdm(range(25)):
    model.train()
    trn_running_loss = 0.0
    trn_size = len(trn_dataloader)
    for X, y in trn_dataloader:
        optimizer.zero_grad()

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        trn_running_loss += loss.item()
    trn_losses.append(trn_running_loss / trn_size)

    model.eval()
    val_running_loss = 0.0
    val_size = len(val_dataloader)
    with torch.no_grad():
        for X, y in val_dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)
            val_running_loss += loss.item()
    val_losses.append(val_running_loss / val_size)

    print(f'epoch: [{epoch + 1:<4}]   training loss: {trn_losses[-1]:.5f}   validation loss: {val_losses[-1]:.5f}')
print('Finished Training')

plt.plot(torch.Tensor(val_losses).cpu())
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

with torch.no_grad():
    print(mean([pfbeta(y, model(X), rsna.class_wt[1]) for X, y in val_dataloader]))

kernel = PolynomialCountSketch(degree=3)
X_train = kernel.fit_transform(rsna.X)

sgd = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=0.0001, class_weight='balanced'))
sgd.fit(X_train, rsna.y)
print(pfbeta(rsna.y, sgd.predict(X_train), rsna.class_wt[1]))

clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, rsna.y)
print(pfbeta(rsna.y, clf.predict(X_train), rsna.class_wt[1]))

rfc = RandomForestClassifier(max_depth=3, class_weight='balanced')
rfc.fit(X_train, rsna.y)
print(pfbeta(rsna.y, rfc.predict(X_train), rsna.class_wt[1]))

mlp = MLPClassifier(hidden_layer_sizes=(10,), early_stopping=True)
mlp.fit(X_train, rsna.y)
print(pfbeta(rsna.y, mlp.predict(X_train), rsna.class_wt[1]))

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
cnb = ComplementNB(force_alpha=True)
cnb.fit(X_scaled, rsna.y)
print(pfbeta(rsna.y, cnb.predict(X_scaled), rsna.class_wt[1]))

width = 0.1
row_names = [m.__class__.__name__ for m in metric[:-1]]
clf_metric = [m(torch.as_tensor(rsna.y).cuda(), torch.as_tensor(clf.predict(X_train)).cuda()).item() for m in metric[:-1]]
sgd_metric = [m(torch.as_tensor(rsna.y).cuda(), torch.as_tensor(sgd.predict(X_train)).cuda()).item() for m in metric[:-1]]
rfc_metric = [m(torch.as_tensor(rsna.y).cuda(), torch.as_tensor(rfc.predict(X_train)).cuda()).item() for m in metric[:-1]]
cnb_metric = [m(torch.as_tensor(rsna.y).cuda(), torch.as_tensor(cnb.predict(X_train)).cuda()).item() for m in metric[:-1]]
mlp_metric = [m(torch.as_tensor(rsna.y).cuda(), torch.as_tensor(mlp.predict(X_train)).cuda()).item() for m in metric[:-1]]
row_names.append('ProbabilisticF1')
clf_metric.append(pfbeta(rsna.y, clf.predict(X_train)))
sgd_metric.append(pfbeta(rsna.y, sgd.predict(X_train)))
rfc_metric.append(pfbeta(rsna.y, rfc.predict(X_train)))
cnb_metric.append(pfbeta(rsna.y, cnb.predict(X_train)))
mlp_metric.append(pfbeta(rsna.y, mlp.predict(X_train)))
plt.bar(np.arange(len(row_names)) + width * -2, clf_metric, width=width, label='LogisticRegression')
plt.bar(np.arange(len(row_names)) + width * -1, sgd_metric, width=width, label='SupportVectorClassifier')
plt.bar(np.arange(len(row_names)) + width * 0, rfc_metric, width=width, label='RandomForestClassifier')
plt.bar(np.arange(len(row_names)) + width * 1, cnb_metric, width=width, label='ComplementNaiveBayes')
plt.bar(np.arange(len(row_names)) + width * 2, mlp_metric, width=width, label='MultiLayerPerceptron')
plt.xticks(np.arange(len(row_names)), row_names, rotation=45)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
plt.show()
