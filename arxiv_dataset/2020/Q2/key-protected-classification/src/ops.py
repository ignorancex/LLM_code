import numpy as np
import torch
import torch.nn as nn
np.set_printoptions(linewidth=150, edgeitems=100)

PRINT_CONFMAT = False
LOG_INTERVAL = 0
GEN_GRAD_NORM_LIMIT = 0.
GEN_MIN_ACC = 0.75
GEN_BUFFER_SIZE = 15
N_MAX_CLASSES = 50


def train_clf(args, model, train_loader, crit, optimizer, n_epochs=1):
    """
    Trains the classification model with one epoch of training data.
    """

    model.train()
    for p in model.parameters(): p.requires_grad = True

    avg_loss = 0.
    n_samples = 0

    conf_mat = torch.zeros(N_MAX_CLASSES, N_MAX_CLASSES, dtype=torch.long, device=args.device)

    for _ in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)

            optimizer.zero_grad()
            loss = crit(output, target)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() * data.size(0)
            conf_mat += compute_confmat(output, target)
            n_samples += data.size(0)

    avg_loss /= n_samples
    print("n_samples:{}, avg_loss:{}".format(n_samples, avg_loss), flush=True)

    # save the conf mat
    conf_mat = conf_mat.cpu().numpy()
    with open(args.output_dir + "/confmat_clf.txt", "a") as fid:
        if args.dataset == "olivetti":
            np.savetxt(fid, conf_mat, fmt="%2d")
        else:
            np.savetxt(fid, conf_mat, fmt="%4d")
        fid.write("\n")
        fid.write("*" * 100 + "\n")
        fid.write("\n")

    if PRINT_CONFMAT:
        print("confusion matrix obtained during training:", flush=True)
        print(conf_mat, flush=True)


def test_clf(args, model, test_loader, crit):
    """
    Tests the classification model over the test set.
    """

    model.eval()
    loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            # loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            loss += crit(output, target).item() * data.shape[0]  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
        loss, correct, len(test_loader.dataset), acc), flush=True)

    return loss, acc


def train_gen(args, clf, gen, c_attack, crit, optimizer, gen_timeout_step):
    """
    Trains the generator by targeting the class c_attack logits of the classification model.
    """

    gen.train()
    clf.train()

    buffer_acc = np.zeros([GEN_BUFFER_SIZE], dtype=np.float32)

    total_loss = 0.
    total_correct = 0
    n_samples = 0
    n_steps = 0
    batch_size = 256

    # keep confusion matrix obtained during training
    conf_mat = torch.zeros(N_MAX_CLASSES, N_MAX_CLASSES, dtype=torch.long, device=args.device)

    # label the fake images as c_attack
    label = torch.zeros(batch_size, dtype=torch.long, device=args.device)
    label.fill_(c_attack)

    loop = True
    while loop:

        # Generate batch of latent vectors
        noise = torch.randn(batch_size, gen.z_dim, 1, 1, device=args.device)

        # Generate fake image batch with gen
        image = gen(noise)

        # Calculate clf's loss on the all-fake batch
        # gen will be trained to reduce this loss
        output = clf(image)
        loss = crit(output, label)
        optimizer.zero_grad()
        loss.backward()
        if GEN_GRAD_NORM_LIMIT > 0.:
            for param in gen.parameters():
                if param.grad is None: continue
                if param.grad.data.norm() <= GEN_GRAD_NORM_LIMIT: continue
                param.grad.data.mul_(GEN_GRAD_NORM_LIMIT / param.grad.data.norm())
        optimizer.step()

        with torch.no_grad():
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            correct = (pred == c_attack).sum().item()
            total_correct += correct

            conf_mat += compute_confmat(output, label)

        buffer_acc[n_steps % GEN_BUFFER_SIZE] = correct / batch_size
        if np.all(buffer_acc >= GEN_MIN_ACC) or (n_steps + 1) == gen_timeout_step:
            loop = False

        total_loss += loss.item()
        n_samples += batch_size
        n_steps += 1

    print("generator training: "
          "n_steps:{}, ".format(n_steps),
          "avg-loss:{}, ".format(total_loss / n_samples),
          "avg-acc:{:.1f} ({}/{}), ".format(100. * total_correct / n_samples, total_correct, n_samples),
          "buffer-acc:{:.1f}".format(100. * buffer_acc.mean()),
          flush=True)

    # save the conf mat
    conf_mat = conf_mat.cpu().numpy()
    with open(args.output_dir + "/confmat_gen.txt", "a") as fid:
        if args.dataset == "olivetti":
            np.savetxt(fid, conf_mat, fmt="%2d")
        else:
            np.savetxt(fid, conf_mat, fmt="%4d")
        fid.write("\n")
        fid.write("*" * 100 + "\n")
        fid.write("\n")


def draw_samples_from(gen, device, N=128, rescale=False):
    """
    Draws samples from the generator network.
    If normalize is True, image pixels are rescaled to [0, 255].
    """
    gen.eval()

    with torch.no_grad():
        noise = torch.randn(N, gen.z_dim, 1, 1, device=device)
        image = gen(noise)

    if rescale:
        image += 1.0
        image /= 2.0
        image *= 255.
        image = torch.clamp(image, 0., 255.).byte().cpu()

    return image


def compute_confmat(y_pred, y):
    """
    Computes confusion matrix given model logits and ground-truth labels.
    This is actually the "update" function of ConfusionMatrix in ignite.metrics.
    """
    # target is (batch_size, ...)
    y_pred = torch.argmax(y_pred, dim=1).flatten()
    y = y.flatten()

    target_mask = (y >= 0) & (y < N_MAX_CLASSES) & (y_pred >= 0) & (y_pred < N_MAX_CLASSES)
    y = y[target_mask]
    y_pred = y_pred[target_mask]

    indices = N_MAX_CLASSES * y + y_pred
    m = torch.bincount(indices, minlength=N_MAX_CLASSES ** 2).reshape(N_MAX_CLASSES, N_MAX_CLASSES)
    return m


def cosine_distance_loss(cosine_sims, labels):
    """
    Args:
        cosine_sims: th.tensor [batch_size, n_classes],
                     cosine similarity values of a batch of samples for a list of classes
        labels: th.tensor [batch_size],
                labels of the samples in the batch
    Returns:
        th.tensor[1], average cosine distance loss
    """
    loss_per_class = 1 - cosine_sims
    loss_gt_class = torch.gather(loss_per_class, 1, labels[:, None])
    loss = loss_gt_class.mean()
    return loss


def approximate_key(key, epsilon):
    """
    args:
        key: th.tensor ([key_dim])
        epsilon: float
    """
    print("approximating a key for epsilon:{}".format(epsilon), flush=True)

    # key that will be optimized to make it approximately epsilon close to the given key
    apprx_key = torch.nn.Parameter(
        torch.randn(key.shape, device=key.device), True)

    crit = lambda x,y : ((x - y).norm() - epsilon).clamp(min=0.)
    l2_dist = lambda x,y : (x - y).norm()
    opt = torch.optim.Adam([apprx_key], 3e-4)

    it = 0
    while True:
        it += 1

        loss = crit(apprx_key, key)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            dist = l2_dist(apprx_key, key)

        if it % 1000 == 0 or loss.item() == 0.:
            print ('iteration: %010d, l2-dist:%f, loss:%f' % (it, dist, loss.item()), flush=True)
            if loss.item() == 0.: break

    return apprx_key.data


def set_learning_rate(optimizer, milestones, gamma, epoch):
    """
    Sets the learning rate of the optimizer based on milestones and gamma parameters.
    """
    assert len(optimizer.param_groups) == 1
    pg = optimizer.param_groups[0]
    lr = pg["lr"]
    if epoch in milestones:
        lr *= gamma
    pg["lr"] = lr
    print("learning rate is set to: {}".format(optimizer.param_groups[0]["lr"]), flush=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
