from train import Trainer
from DataUtil import Datasetdrug1, Datasetdrug2, GetData, Motif1, Motif2
import torch
from pytorch_metric_learning.losses import NTXentLoss
from model import Encoder, Classifier

# Check Motif structure code as the vocabulary is obtained on testing data
#  ************************************************************************

epochs = 0
batch_size = 128
data_split = 0  # 0-4

dataset_Drug1 = Datasetdrug1("dataset/DrugComb")
dataset_Drug2 = Datasetdrug2("dataset/DrugComb")

# Change data split in data
dataset_Motif1 = Motif1("dataset/DrugComb")
dataset_Motif2 = Motif2("dataset/DrugComb")

Train_Test_Split = GetData(dataset_Drug1, dataset_Drug2, dataset_Motif1, dataset_Motif2, data_split)
TrainDataset, TestDataset = Train_Test_Split.data_loader(batch_size)

encoder = Encoder()
classifier = Classifier()

optimizer_con = torch.optim.Adam(encoder.parameters(), lr=0.0001)
optimizer_class = (torch.optim.Adam(classifier.parameters(), lr=0.0001))
loss_classifier = torch.nn.BCELoss()
loss_con = NTXentLoss(temperature=0.1)
ModelTrainer = Trainer(TrainDataset, TestDataset, encoder, classifier, epochs, optimizer_con,
                       optimizer_class,
                       loss_con, loss_classifier, batch_size)
ModelTrainer.run()
