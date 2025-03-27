from model.vgg_backbone import VGG_Reconstructor
from model.new_vgg import vgg, VGG

def train(epoch:int, reconstructor:VGG_Reconstructor, vgg:VGG, optimizer, train_loader, criterion, args:dict):
    print(print('\nEpoch:[%d/%d]' % (epoch, args['epoch'])))
    reconstructor.train()
    loss, correct, total = 0, 0, 0
    for index, data in enumerate(train_loader):
        image = data['image'].to(args['device'])
        label = data['label'].to(args['device'])
        optimizer.zero_grad()
        pred, feature = vgg(image)
        feature.to(args['device'])

