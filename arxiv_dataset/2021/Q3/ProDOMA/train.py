from dataset import *
from utils import *

from models.PepCNN import *
# from models.DnaCNN import *
# from models.RNN import *
# from models.SingleFrameCNN import *
# from models.Transformer import Transformer
# from models.openset import *

criterion = nn.CrossEntropyLoss().cuda()


def train(train_loader, val_loader, test_loader, model, optimizer, args):
    model.cuda()

    steps = 0
    best_acc = 0
    best_loss = float('inf')

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    train_info = {'epoch': [], 'train_loss': [], 'val_loss': [], 'metric': [], 'best': [], 'test_loss': [],
                  'test_acc': []}

    print(
        'epoch |   lr    |    %        |  loss  |  avg   |val loss| top1  |  top3   |  best  |test loss| top1  | time | save |')
    bg = time.time()
    train_iter = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        losses = []
        train_loss = 0
        last_val_iter = 0
        current_lr = get_lrs(optimizer)
        for batch_idx, batch in enumerate(train_loader):
            train_iter += 1
            feature, target, _ = batch[0], batch[1], batch[2]
            # print(feature.shape)
            # feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if isinstance(feature, list):
                #   feature contains seqs and positions
                feature[0], feature[1], target = feature[0].cuda(), feature[1].cuda(), target.cuda()
            else:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = data_parallel(model, feature)

            loss = criterion(logit, target)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            train_loss += loss.item()
            losses.append(loss.item())

            print('\r {:4d} | {:.5f} | {:4d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size * (batch_idx + 1), train_loader.num, loss.item(),
                                             train_loss / (train_iter - last_val_iter)), end='')

            if train_iter > 0 and train_iter % args.iter_val == 0:

                top_1, top_3, val_loss, size = validate(val_loader, model)
                test_top_1, tst_top_3, test_loss, _ = validate(test_loader, model)
                _save_ckp = ' '

                if val_loss < best_loss:
                    best_acc = top_1
                    best_loss = val_loss
                    save_checkpoint(args.checkpoint_path, model, optimizer)
                    _save_ckp = '*'

                print(' {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:4s} |'.format(val_loss, top_1,
                                                                                                       top_3, best_acc,
                                                                                                       test_loss,
                                                                                                       test_top_1,
                                                                                                       (
                                                                                                               time.time() - bg) / 60,
                                                                                                       _save_ckp))

                train_info['epoch'].append(args.batch_size * (batch_idx + 1) / train_loader.num + epoch)
                train_info['train_loss'].append(train_loss / (batch_idx + 1))
                train_info['val_loss'].append(val_loss)
                train_info['metric'].append(top_1)
                train_info['best'].append(best_acc)
                train_info['test_loss'].append(test_loss)
                train_info['test_acc'].append(test_top_1)

                log_df = pd.DataFrame(train_info)
                log_df.to_csv(args.checkpoint_path + '.csv')

                train_loss = 0
                last_val_iter = train_iter

                model.train()

    log_df = pd.DataFrame(train_info)
    log_df.to_csv(args.checkpoint_path + '.csv')
    print("Best accuracy is {:.4f}".format(best_acc))


def validate(data_loader, model):
    model.eval()
    corrects = []
    losses = []
    for batch in data_loader:
        feature, target = batch[0], batch[1]
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if isinstance(feature, list):
            #   feature contains seqs and positions
            feature[0], feature[1], target = feature[0].cuda(), feature[1].cuda(), target.cuda()
        else:
            feature, target = feature.cuda(), target.cuda()

        with torch.no_grad():
            logit = model(feature)
            loss = criterion(logit, target)

            losses.append(loss.item())
            correct = metric(logit, target)
            corrects.append(correct.data.cpu().numpy())

    correct = np.concatenate(corrects)
    correct = correct.mean(0)
    loss = np.mean(losses)
    top = [correct[0], correct[0] + correct[1], correct[0] + correct[1] + correct[2]]
    size = len(data_loader.dataset)
    return top[0], top[2], loss, size


if __name__ == '__main__':
    args = argparser()
    args.vocab_size = 21
    print(args)

    EMBED_MODEL = ['PepCNNDeepEmbed']
    POS_MODEL = ['Transformer']
    if args.model in EMBED_MODEL:
        args.use_embed = True
    else:
        args.use_embed = False
    if args.model in POS_MODEL:
        args.position = True
    else:
        args.position = False


    train_data = DnapepDataset(
        file_path=args.train_file,
        type='test', use_embed=args.use_embed)
    # test_data = PepseqDataset(file_path="/home/dunan/Documents/DeepFam_data/GPCR/cv_1/test.txt")
    val_data = DnapepDataset(
        file_path=args.valid_file,
        type='test', use_embed=args.use_embed)
    # test_data = DnapepDataset(file_path="/home/dunan/Documents/DeepFam_data/GPCR_cds_pbsim/test_first40k.txt",
    # type='test')
    test_data = DnapepDataset(
        file_path=args.test_file,
        type='test', use_embed=args.use_embed)

    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    train_loader.num = len(train_data)
    val_loader = data.DataLoader(val_data, batch_size=256, num_workers=4)
    test_loader = data.DataLoader(test_data, batch_size=256, num_workers=4)

    # model = eval(args.model)(num_class=args.num_classes, kernel_nums=args.num_filters, kernel_sizes=args.filter_sizes,
    #                          dropout=args.dropout, num_fc=args.num_hidden)
    model = eval(args.model)(args)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.l2)

    train(train_loader, val_loader, test_loader, model, optimizer, args)
