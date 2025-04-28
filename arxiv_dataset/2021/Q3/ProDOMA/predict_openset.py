from utils import *
from dataset import *
from models.PepCNN import *

MODEL_WO_LOAD = ['PosWiseCNN', 'PepCNNFromSingle']


def predict(args):
    if args.threshold is None:
        threshold = np.array([0.0] * args.num_classes)

    else:
        threshold = np.load(args.threshold)

    print(threshold)

    model = eval(args.model)(args)
    if args.model not in MODEL_WO_LOAD:
        load_checkpoint(args.checkpoint_path, model)
    model.cuda()
    model.eval()

    probs = []
    topks = []
    y_pred = []
    y_true = []

    y_score = []

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

            scores, preds = torch.max(prob, 1)
            scores, preds = scores.tolist(), preds.tolist()
            for i, pred in enumerate(preds):
                y_score.append(scores[i])
                if scores[i] > threshold[pred]:
                    y_pred.append(pred)

                else:
                    y_pred.append(-1)

            y_true += target.tolist()
            logit_5, top5 = torch.topk(prob.data.cpu(), args.topk)
            for i, l in enumerate(logit_5):
                probs.append(l.numpy())
                topks.append(top5[i].numpy())

    size = len(data_loader.dataset)
    accuracy = accuracy_score(y_true, y_pred)
    print("acc: {:.4f}%({}/{})".format(100 * accuracy, int(accuracy * size), size))

    # print(y_true)
    # print(y_pred)
    idx = np.argsort(y_score)
    # print(np.array(y_true)[idx])

    true = 0
    for y in y_true:
        if y != -1:
            true += 1

    tp = 0
    p = 0
    for i, y in enumerate(y_pred):
        if y != -1:
            p += 1
            if y == y_true[i]:
                tp += 1
    print(tp)
    print(p)
    print(true)
    recall = tp / true
    precision = tp / p
    print(f'recall is {recall}')
    print(f'precision is {precision}')
    print(f'f1 is {2*recall * precision / (recall + precision)}')

    if args.predict_file:
        df = pd.read_csv(args.test_file, sep='\t', header=None)
        df["probs"] = probs
        df["topk"] = topks
        df["pred"] = y_pred
        df.to_csv(args.predict_file, columns=[2, 0, "pred", "topk", "probs"])


if __name__ == '__main__':
    args = argparser()
    predict(args)