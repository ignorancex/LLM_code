from tqdm import tqdm
import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision
import os


class Trainer:
    def __init__(self, train_loader, test_loader, encoder, classifier, epochs, optimizer_encoder, optimizer_classifier, loss_con, loss_classifier, batch_size):
        self.TrainDataset_Drug1, self.TrainDataset_Drug2, self.TrainDataset_Motif1, self.TrainDataset_Motif2 = train_loader
        self.TestDataset_Drug1, self.TestDataset_Drug2, self.TestDataset_Motif1, self.TestDataset_Motif2 = test_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder = encoder.to(self.device)
        self.classifier = classifier.to(self.device)
        self.epochs = epochs
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_classifier = optimizer_classifier
        self.loss_class = loss_classifier
        self.loss_con = loss_con
        self.batch_size = batch_size

    def train_encoder_epoch(self):
        running_loss = 0.0
        step = 0
        self.encoder = self.encoder.float()
        self.encoder.train(mode=True)
        for _, batch in enumerate(tqdm(zip(self.TrainDataset_Drug1, self.TrainDataset_Drug2, self.TrainDataset_Motif1, self.TrainDataset_Motif2))):
            batch1 = batch[0]
            batch2 = batch[1]
            batch3 = batch[2]
            batch4 = batch[3]
            batch1 = batch1.to(self.device)
            batch2 = batch2.to(self.device)
            batch3 = batch3.to(self.device)
            batch4 = batch4.to(self.device)

            self.optimizer_encoder.zero_grad()
            pred = self.encoder(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.coord, batch1.batch, batch2.x,
                                batch2.edge_index, batch2.edge_attr, batch2.coord, batch2.batch, batch3.x.float(),  # Messed up edge_attr -> Wrote as adge_attr
                                batch3.edge_index.type(torch.int64),
                                batch3.batch, batch4.x.float(), batch4.edge_index.type(torch.int64), batch4.batch,
                                batch2.Expression)
            loss = self.loss_con(pred, batch2.Synergy)

            loss.backward()
            self.optimizer_encoder.step()
            running_loss += loss.item()
            step += 1
        return running_loss/step

    def train_epoch(self):
        running_loss = 0.0
        step = 0
        self.encoder = self.encoder.float()
        self.encoder.train(mode=False)
        self.classifier = self.classifier.float()
        self.classifier.train(mode=True)
        ROC_AUC = 0.0
        for _, batch in enumerate(tqdm(zip(self.TrainDataset_Drug1, self.TrainDataset_Drug2, self.TrainDataset_Motif1, self.TrainDataset_Motif2))):
            batch1 = batch[0]
            batch2 = batch[1]
            batch3 = batch[2]
            batch4 = batch[3]
            batch1 = batch1.to(self.device)
            batch2 = batch2.to(self.device)
            batch3 = batch3.to(self.device)
            batch4 = batch4.to(self.device)

            self.optimizer_classifier.zero_grad()
            with torch.no_grad():
                pred = self.encoder(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.coord, batch1.batch, batch2.x,
                                    batch2.edge_index, batch2.edge_attr, batch2.coord, batch2.batch, batch3.x.float(),
                                    batch3.edge_index.type(torch.int64),
                                    batch3.batch, batch4.x.float(), batch4.edge_index.type(torch.int64), batch4.batch,
                                    batch2.Expression)
            pred = self.classifier(pred)
            loss = self.loss_class(pred, batch2.Synergy)

            loss.backward()
            self.optimizer_classifier.step()
            running_loss += loss.item()
            step += 1
            auroc = AUROC(task="binary")
            ROC_AUC += auroc(pred, batch2.Synergy)
        return running_loss/step, ROC_AUC/step

    def test(self):
        running_loss = 0.0
        step = 0
        ROC_AUC = 0.0
        ACC = 0.0
        AUPR = 0.0
        self.encoder = self.encoder.float()
        self.encoder.train(mode=False)
        self.classifier = self.classifier.float()
        self.classifier.train(mode=False)
        with torch.no_grad():
            for _, batch in enumerate(tqdm(zip(self.TestDataset_Drug1, self.TestDataset_Drug2, self.TestDataset_Motif1, self.TestDataset_Motif2))):
                batch1 = batch[0]
                batch2 = batch[1]
                batch3 = batch[2]
                batch4 = batch[3]
                batch1 = batch1.to(self.device)
                batch2 = batch2.to(self.device)
                batch3 = batch3.to(self.device)
                batch4 = batch4.to(self.device)

                pred = self.encoder(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.coord, batch1.batch, batch2.x,
                                    batch2.edge_index, batch2.edge_attr, batch2.coord, batch2.batch, batch3.x.float(),
                                    batch3.edge_index.type(torch.int64),
                                    batch3.batch, batch4.x.float(), batch4.edge_index.type(torch.int64), batch4.batch,
                                    batch2.Expression).to(self.device)
                pred = self.classifier(pred)
                loss = self.loss_class(pred, batch2.Synergy)

                running_loss += loss.item()
                step += 1
                auroc = AUROC(task="binary").to(self.device)
                AP = AveragePrecision(task='binary').to(self.device)
                acc = Accuracy(task='binary').to(self.device)
                ROC_AUC += auroc(pred, batch2.Synergy)
                AUPR += AP(pred, batch2.Synergy.type(torch.int64))
                ACC += acc(pred, batch2.Synergy)
        return running_loss/step, ROC_AUC/step, AUPR/step, ACC/step

    def run(self):
        print("Training Encoder")
        for i in range(self.epochs):
            loss_train = self.train_encoder_epoch()
            print(f"Epoch {i}: Encoder Loss: " + str(loss_train))
            torch.save(self.encoder.state_dict(), os.path.join("encoder_weights", 'epoch-{}.pt'.format(i)))

        print("Training Classifier")
        for i in range(25):
            loss_train, auc = self.train_epoch()
            print(f"Epoch {i}: Train Loss: " + str(loss_train) + ", Train AUCROC: " + str(auc))
            loss_test, auc_test, AUPR_test, accuracy = self.test()
            print(f"Epoch {i}: Test Loss: " + str(loss_test) + ", Test AUCROC: " + str(auc_test) + ", Test AUPR: " + str(AUPR_test) + ", Test Accuracy: "
                                            + str(accuracy))
            torch.save(self.classifier.state_dict(), os.path.join("encoder_weights", 'epoch-classifier-{}.pt'.format(i)))