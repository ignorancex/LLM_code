import time
import torch
import numpy as np

from torch.utils.data import dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(output, target, topk=(1,)):
    """Computes the pervision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def eval_model(model, criterion, dataloaders,
               dataset_sizes, writer=None, epoch=0, class_acc=None):

    since = time.time()

    best_acc = 0.0
    if isinstance(dataloaders['train'].dataset, dataset.Subset):

        num_classes = len(dataloaders['train'].dataset.dataset.classes)
    else:
        num_classes = len(dataloaders['train'].dataset.classes)

    # Each epoch has a training and validation phase
    for phase in ['val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        acc1_sum = 0
        acc5_sum = 0
        count = 0
        running_class_corrects = np.zeros(num_classes)
        running_class_total = np.zeros(num_classes)

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(phase == 'train'):
                activations, outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                acc1_sum += acc1
                acc5_sum += acc5
                count += 1
                loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            c = (preds == labels.data).squeeze()
            for i in range(labels.shape[0]):
                label = labels.data[i]
                running_class_corrects[label] += c[i]
                running_class_total[label] += 1

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_class_acc = running_class_corrects / running_class_total
        epoch_acc1 = acc1_sum / count
        epoch_acc5 = acc5_sum / count

        if writer:
            writer.add_scalar('Loss/Eval', epoch_loss, epoch)

            if class_acc is not None:
                writer.add_histogram(
                    'Class_Acc_Delta/Eval',
                    epoch_class_acc - class_acc,
                    epoch
                )

            writer.add_histogram('Class_Acc/Eval', epoch_class_acc, epoch)
            writer.add_scalar('Top1/Eval', epoch_acc1, epoch)
            writer.add_scalar('Top5/Eval', epoch_acc5, epoch)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc1))

        # deep copy the model
        if phase == 'val' and epoch_acc1 > best_acc:
            best_acc = epoch_acc1

    time_elapsed = time.time() - since
    print('Eval complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('val Acc: {:4f}'.format(best_acc))

    return best_acc


def first_eval(model, criterion, dataloaders,
               dataset_sizes, writer=None, epoch=0):
    since = time.time()

    best_acc = 0.0
    if isinstance(dataloaders['train'].dataset, dataset.Subset):
        num_classes = len(dataloaders['train'].dataset.dataset.classes)
    else:
        num_classes = len(dataloaders['train'].dataset.classes)

    ground_labels = np.array([])
    predictions = np.array([])
    # Each epoch has a training and validation phase
    for phase in ['val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        acc1_sum = 0
        acc5_sum = 0
        count = 0
        running_class_corrects = np.zeros(num_classes)
        running_class_total = np.zeros(num_classes)

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(phase == 'train'):
                activations, outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                acc1_sum += acc1
                acc5_sum += acc5
                count += 1
                loss = criterion(outputs, labels)

            ground_labels = np.append(ground_labels, labels.cpu().data)
            predictions = np.append(predictions, preds.cpu().data)
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            c = (preds == labels.data).squeeze()
            for i in range(labels.shape[0]):
                label = labels.data[i]
                running_class_corrects[label] += c[i]
                running_class_total[label] += 1

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_class_acc = running_class_corrects / running_class_total
        epoch_acc1 = acc1_sum / count
        epoch_acc5 = acc5_sum / count

        if writer:
            writer.add_scalar('Loss/Eval', epoch_loss, epoch)
            writer.add_histogram('Class_Acc/Eval', epoch_class_acc, epoch)
            writer.add_scalar('Top1/Eval', epoch_acc1, epoch)
            writer.add_scalar('Top5/Eval', epoch_acc5, epoch)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc1))

        # deep copy the model
        if phase == 'val' and epoch_acc1 > best_acc:
            best_acc = epoch_acc1

    time_elapsed = time.time() - since
    print('Eval complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return epoch_class_acc


def train_student_kd(student, teacher, criterion, inner_loss,
                     dataloaders, dataset_sizes, optimizer,
                     scheduler, writer=None, epoch=0, phase='train', alpha=1):
    since = time.time()

    teacher.eval()

    if phase == 'train' or phase == 'ss_train':
        student.train()  # Set student to training mode
    else:
        student.eval()   # Set student to evaluate mode

    if isinstance(dataloaders['train'].dataset, dataset.Subset):
        num_classes = len(dataloaders['train'].dataset.dataset.classes)
    else:
        num_classes = len(dataloaders['train'].dataset.classes)
    running_loss = 0.0
    running_corrects = 0
    acc1_sum = 0
    acc5_sum = 0
    count = 0
    running_class_corrects = np.zeros(num_classes)
    running_class_total = np.zeros(num_classes)

    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train' or phase == 'ss_train'):
            student_act, outputs = student(inputs)
            teacher_act, teacher_outputs = teacher(inputs)
            _, preds = torch.max(outputs, 1)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            acc1_sum += acc1
            acc5_sum += acc5
            count += 1
            act_losses = [inner_loss(s_a, t_a)
                          for s_a, t_a in zip(student_act, teacher_act)]
            act_loss = torch.stack(act_losses, dim=0).sum()
            if phase == 'ss_train':
                loss = act_loss
            else:
                loss = alpha * criterion(outputs, labels)
                loss += act_loss

            # backward + optimize only if in training phase
            if phase == 'train' or phase == 'ss_train':
                loss.backward()
                optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        c = (preds == labels.data).squeeze()
        for i in range(labels.shape[0]):
            label = labels.data[i]
            running_class_corrects[label] += c[i]
            running_class_total[label] += 1

    if phase == 'train':
        scheduler.step()

    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_class_acc = running_class_corrects / running_class_total
    epoch_acc1 = acc1_sum / count
    epoch_acc5 = acc5_sum / count

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc1))

    if writer:
        writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
        writer.add_histogram(f'Class_Acc/{phase}', epoch_class_acc, epoch)
        writer.add_scalar(f'Top1/{phase}', epoch_acc1, epoch)
        writer.add_scalar(f'Top5/{phase}', epoch_acc5, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print(phase + ' Acc: {:4f}'.format(epoch_acc1))

    return student


def train_baseline(student, criterion, dataloaders, dataset_sizes, optimizer,
                   scheduler, writer=None, epoch=0, phase='train'):
    since = time.time()

    if phase == 'train':
        student.train()  # Set student to training mode
    else:
        student.eval()   # Set student to evaluate mode

    if isinstance(dataloaders['train'].dataset, dataset.Subset):
        num_classes = len(dataloaders['train'].dataset.dataset.classes)
    else:
        num_classes = len(dataloaders['train'].dataset.classes)
    running_loss = 0.0
    running_corrects = 0
    acc1_sum = 0
    acc5_sum = 0
    count = 0
    running_class_corrects = np.zeros(num_classes)
    running_class_total = np.zeros(num_classes)

    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            _, outputs = student(inputs)
            _, preds = torch.max(outputs, 1)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            acc1_sum += acc1
            acc5_sum += acc5
            count += 1

            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        c = (preds == labels.data).squeeze()
        for i in range(labels.shape[0]):
            label = labels.data[i]
            running_class_corrects[label] += c[i]
            running_class_total[label] += 1

    if phase == 'train':
        scheduler.step()

    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_class_acc = running_class_corrects / running_class_total
    epoch_acc1 = acc1_sum / count
    epoch_acc5 = acc5_sum / count

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc1))

    if writer:
        writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
        writer.add_histogram(f'Class_Acc/{phase}', epoch_class_acc, epoch)
        writer.add_scalar(f'Top1/{phase}', epoch_acc1, epoch)
        writer.add_scalar(f'Top5/{phase}', epoch_acc5, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print(phase + ' Acc: {:4f}'.format(epoch_acc1))

    return student


def class_report(model, dataloaders, dataset_sizes, writer=None, epoch=0):

    ground_labels = np.array([])
    predictions = np.array([])
    # Each epoch has a training and validation phase
    for phase in ['val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            ground_labels = np.append(ground_labels, labels.cpu().data)
            predictions = np.append(predictions, preds.cpu().data)
            # statistics

    return ground_labels, predictions
