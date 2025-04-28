from utils import *
from dataset import *
from models.PepCNN import *
from models.RNN import *
from models.SingleFrameCNN import PosWiseCNN, PepCNNFromSingle

MODEL_WO_LOAD = ['PosWiseCNN', 'PepCNNFromSingle']

def predict(args):
    model = eval(args.model)(args)
    if args.model not in MODEL_WO_LOAD:
        load_checkpoint(args.checkpoint_path, model)
    model.cuda()
    model.eval()

    probs = []
    topks = []
    y_pred = []
    y_true = []

    predict_data = DnapepDataset(args.test_file, type='test', shuffle=False)
    data_loader = data.DataLoader(predict_data, batch_size=args.batch_size)

    corrects = 0
    for batch in tqdm.tqdm(data_loader):
        feature, target, _ = batch[0], batch[1], batch[2]
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        feature, target = feature.cuda(), target.cuda()

        with torch.no_grad():
            logit = data_parallel(model, feature)
            prob = F.softmax(logit, 1)

            y_pred += torch.max(prob, 1)[1].tolist()
            y_true += target.tolist()
            corrects += (torch.max(prob, 1)
                         [1].view(target.size()).data == target.data).sum()
            logit_5, top5 = torch.topk(prob.data.cpu(), args.topk)
            for i, l in enumerate(logit_5):
                probs.append(l.tolist())
                topks.append(top5[i].tolist())

    size = len(data_loader.dataset)
    accuracy = 100 * corrects.data.cpu().numpy() / size
    print("acc: {:.4f}%({}/{})".format(accuracy, corrects, size))
    cls_label = list(set(y_true))
    cls_label.sort()
    # cm = confusion_matrix(y_true, y_pred)
    # plot_confusion_matrix(cm, target_names=cls_label)

    if args.predict_file:
        df = pd.read_csv(args.test_file, sep='\t', header=None)
        df["probs"] = probs
        df["topk"] = topks
        df.to_csv(args.predict_file, columns=[2, 0, "topk", "probs"])

        print(df.head())

        base = os.path.basename(args.predict_file)
        parent_dir = os.path.dirname(args.predict_file)
        file_path = os.path.join(parent_dir, os.path.splitext(base)[0] + '_probs.npy')
        np.save(file_path, probs)

        file_path = os.path.join(parent_dir, os.path.splitext(base)[0] + '_topk.npy')
        np.save(file_path, topks)


if __name__ == '__main__':
    args = argparser()
    predict(args)
