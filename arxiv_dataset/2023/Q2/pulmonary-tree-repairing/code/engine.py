from etaprogress.progress import ProgressBar
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from ptflops import get_model_complexity_info
from utils import *
# import matplotlib.pyplot as plt
# from scipy import stats
from evaluate import accuracy
from loss import KeypointMSELoss
from model.UNET import UNet3D
from dataset import Dataset_Train, Dataset_Test
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.backends.cudnn.benchmark = True

class Engine:
    def __init__(self, config):

        # Model Configuration
        self.root = config['root']
        self.checkpoint_path = config['checkpoint_path']
        self.log_path = config['log_path']
        self.val_interval = int(config['val_interval'])
        self.batch_size = int(config['batch_size'])
        self.bar_len = int(config['bar_len'])
        self.sub_size = int(config['sub_size'])
        self.step = int(config['step'])
        self.optimizer_name = config['optimizer']
        self.model_name = config['model']

        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        # Preparing model(s) for GPU acceleration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Choose model
        self.model = UNet3D(in_channels=2, num_classes=2)
        self.model = nn.DataParallel(self.model).to(self.device)

        # Initiating Training Parameters(for step)
        self.currentSteps = 0
        self.totalSteps = self.step
        self.best_ap = 0
        self.best_step =0
        self.val_best_ap = 0

        # Optimizers
        if self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.5, 0.999))
        elif self.optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001,
                                          momentum=0.9,
                                          weight_decay=0.0001,
                                          nesterov=False)

        self.train_dataset, self.val_dataset = None, None

        # loss
        self.criterion = KeypointMSELoss().to(self.device)
        
        # log
        self.writer = SummaryWriter(self.log_path)


    def train_val_loader(self):
        train_list = get_npz_list(os.path.join(self.root,'train'))
        val_list = get_npz_list(os.path.join(self.root,'val_4repro'))

        print("Training Sample number: {}".format(len(train_list)))
        print("Validation Sample number: {}".format(len(val_list)))

        train_dataset = Dataset_Train(train_list, self.sub_size)
        val_dataset = Dataset_Test(val_list)

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                         batch_size=self.batch_size,
                                                         shuffle=True,
                                                         num_workers = 4,
                                                         pin_memory=True
                                                         )

        self.val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                        batch_size=4,
                                                        shuffle=False,
                                                        num_workers = 4,
                                                        pin_memory=True
                                                        )


    def model_training(self, resume_training=True, checkpoint=None):
        batch_time = AverageMeter()
        losses = AverageMeter()
        exp_dis = AverageMeter()
        ap = AverageMeter()
        exp_dis_kp1 = AverageMeter()
        exp_dis_kp2 = AverageMeter()
        if resume_training and checkpoint != None:
            try:
                self.model_load(checkpoint)

            except IOError:
                # print()
                custom_print(Fore.RED + "Would you like to start training from sketch (default: Y): ",
                             text_width=self.bar_len)
                user_input = input() or "Y"
                if not (user_input == "Y" or user_input == "y"):
                    exit()

        
        # Starting Training
        custom_print('Training is about to begin using:' + Fore.YELLOW + '[{}]'.format(self.device).upper(),
                     text_width=self.bar_len)

        self.train_val_loader()

        # Initiating progress bar
        bar = ProgressBar(self.totalSteps, max_width=int(self.bar_len / 2))

        self.model.train()
        # Train
        start_time = time.time()
        scaler = GradScaler()
        while self.currentSteps < self.totalSteps:
            
            for i, (volume, target, target_weight, edge_radius, edge_volume) in enumerate(self.train_loader):
                batch_start_time = time.time()
                volume = volume.type(torch.FloatTensor).cuda(non_blocking=True)
                target = target.type(torch.FloatTensor).cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)
                batch_size = volume.size(0)

                with autocast(enabled=True):
                    output = self.model(volume)               
                    loss = self.criterion(output, target, target_weight)
                   
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                # measure accuracy and record loss
                
                metric = accuracy(output.detach().cpu().numpy(),
                                  target.detach().cpu().numpy(),
                                  target_weight.detach().cpu().numpy(),
                                  edge_radius.detach().cpu().numpy(),
                                  edge_volume.detach().cpu().numpy(),
                                  mode='train'
                                  )

                losses.update(loss.item(), metric['val_batch_size'])
                ap.update(metric['ap'], metric['val_batch_size'])
                exp_dis.update(metric['exp_dis'], metric['vis_keypoint'])
                exp_dis_kp1.update(metric['exp_dis_kp1'],metric['vis_kp1'])
                exp_dis_kp2.update(metric['exp_dis_kp2'],metric['vis_kp2'])

                # measure elapsed time
                batch_time.update(time.time() - batch_start_time)
                self.currentSteps += 1
                h, m, s = timer(start_time, time.time())
                
                ##########################
                ###### Model Logger ######
                ##########################

                if self.currentSteps % 20 == 0:
                    print('Step:[{}/{}] | Time:{:0>2}:{:0>2}:{:0>2} | Speed:{:.1f} | Loss:{:.7f} | EXP_D:{:.3f} | AP:{:.3f} | EXP_D_KP1:{:.3f} | EXP_D_KP2:{:.3f} '.format(
                        self.currentSteps,self.totalSteps, 
                        h, m, s,
                        batch_size/batch_time.val,
                        losses.val,
                        exp_dis.val,
                        ap.val,
                        exp_dis_kp1.val,
                        exp_dis_kp2.val
                        ))
                  
                    # print('Effective keypoint number:',target_weight.sum().item())

                    self.writer.add_scalar("Training/train_loss",
                                  scalar_value=losses.val,
                                  global_step=self.currentSteps)
                
                    self.writer.add_scalar("Training/train_exp_dis",
                                  scalar_value=exp_dis.val,
                                  global_step=self.currentSteps)

                    self.writer.add_scalar("Training/train_ap",
                                  scalar_value=ap.val,
                                  global_step=self.currentSteps)

                if self.currentSteps % self.val_interval == 0 or self.currentSteps == self.totalSteps:
                    val_ap, val_exp_dis, val_loss = self.model_validation()
                    
                    self.writer.add_scalar("Validation/val_exp_loss",
                                  scalar_value=val_loss,
                                  global_step=self.currentSteps)
                    
                    self.writer.add_scalar("Validation/val_exp_dis",
                                  scalar_value=val_exp_dis,
                                  global_step=self.currentSteps)

                    self.writer.add_scalar("Validation/val_ap",
                                  scalar_value=val_ap,
                                  global_step=self.currentSteps)

                    checkpoint = {
                            'step': self.currentSteps,
                            'stateDict': self.model.module.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'ap': val_ap,
                            'exp_dis': val_exp_dis
                            }

                    if val_ap > self.val_best_ap:
                        self.val_best_ap = val_ap                
                        torch.save(checkpoint, os.path.join(self.checkpoint_path,'model_best.pt'))

                    if self.currentSteps == self.totalSteps:
                        h, m, s = timer(start_time, time.time())
                        print('Total Time:{:0>2}:{:0>2}:{:0>2}'.format(h, m, s))
                        custom_print(Fore.YELLOW + "Training Completed Successfully!", text_width=self.bar_len)
                        torch.save(checkpoint, os.path.join(self.checkpoint_path,'model_final.pt'))
   

    def model_validation(self):
        custom_print('Validation begins...' + Fore.YELLOW + '[{}]'.format(self.device).upper(),
                     text_width=self.bar_len)
        data_len = len(self.val_loader)
        bar_val = ProgressBar(data_len, max_width=self.bar_len)
        losses = AverageMeter()
        batch_time = AverageMeter()
        exp_dis = AverageMeter()
        ap = AverageMeter()
        exp_dis_kp1 = AverageMeter()
        exp_dis_kp2 = AverageMeter()
        
        self.model.eval()

        totalSteps = len(self.val_loader)
        currentSteps = 0

        with torch.no_grad():
            for i, (volume, target, target_weight, edge_radius, edge_volume) in enumerate(self.val_loader):
                batch_start_time = time.time()
                volume = torch.flatten(volume, start_dim=0, end_dim=1) # (b*3,2,80,80,80)
                target = torch.flatten(target, start_dim=0, end_dim=1) # (b*3,2,80,80,80)
                target_weight = torch.flatten(target_weight, start_dim=0, end_dim=1) # (b*3,2,1)
                edge_radius = torch.flatten(edge_radius, start_dim=0, end_dim=1) # (b*3,1)
                edge_volume = torch.flatten(edge_volume, start_dim=0, end_dim=1) # (b*3,1)

                volume = volume.type(torch.FloatTensor).cuda(non_blocking=True)
                target = target.type(torch.FloatTensor).cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                batch_size = volume.size(0)

                output= self.model(volume)

                loss = self.criterion(output, target, target_weight)

                metric = accuracy(output.detach().cpu().numpy(),
                                  target.detach().cpu().numpy(),
                                  target_weight.detach().cpu().numpy(),
                                  edge_radius.detach().cpu().numpy(),
                                  edge_volume.detach().cpu().numpy(),
                                  mode='train'
                                  )

                losses.update(loss.item(), metric['val_batch_size'])
                ap.update(metric['ap'], metric['val_batch_size'])
                exp_dis.update(metric['exp_dis'], metric['vis_keypoint'])
                exp_dis_kp1.update(metric['exp_dis_kp1'],metric['vis_kp1'])
                exp_dis_kp2.update(metric['exp_dis_kp2'],metric['vis_kp2'])

                # measure elapsed time
                batch_time.update(time.time() - batch_start_time)
                currentSteps += 1

                if currentSteps % 10 == 0:
                    print('Step:[{}/{}] | Time:{:.0f} | Loss:{:.5f} | EXP_D:{:.4f} | AP:{:.3f} | EXP_D_KP1:{:.4f} | EXP_D_KP2:{:.4f} '.format(
                        currentSteps,totalSteps, 
                        batch_time.sum,
                        losses.val,
                        exp_dis.val,
                        ap.val,
                        exp_dis_kp1.val,
                        exp_dis_kp2.val
                        ))       


            custom_print(Fore.GREEN + "Validation Completed!", text_width=self.bar_len)
            print('Validation Time:{:.0f}, Loss:{:.5f}, EXP_d:{:.3f}, AP:{:.3f}, EXP_d_Kp1:{:.3f}, EXP_d_Kp2:{:.3f}'.format(
                batch_time.sum,
                losses.avg,
                exp_dis.avg,
                ap.avg,
                exp_dis_kp1.avg, 
                exp_dis_kp2.avg
                ))
            
        self.model.train()

        return ap.avg, exp_dis.avg, losses.avg

    def model_test(self, ckpt_path):
        self.model_load(ckpt_path)
        custom_print('Testing begins...' + Fore.YELLOW + '[{}]'.format(self.device).upper(),
                     text_width=self.bar_len)

        test_list = get_npz_list(os.path.join(self.root,'test'))
        test_dataset = Dataset_Test(test_list)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=8,
                                                  shuffle=False,
                                                  num_workers = 4,
                                                  pin_memory=True
                                                  )

        data_len = len(test_loader)
        bar_val = ProgressBar(data_len, max_width=self.bar_len)
        batch_time = AverageMeter()
        exp_dis = AverageMeter()
        ap = AverageMeter()
        ap_50 = AverageMeter()
        ap_75 = AverageMeter()
        ap_s = AverageMeter()
        ap_m = AverageMeter()
        ap_l = AverageMeter()
        exp_dis_kp1 = AverageMeter()
        exp_dis_kp2 = AverageMeter()
        ap_kp1 = AverageMeter()
        ap_kp2 = AverageMeter()

        self.model.eval()

        totalSteps = len(test_loader)
        currentSteps = 0

        with torch.no_grad():
            for i, (volume, target, target_weight, edge_radius, edge_volume) in enumerate(test_loader):
                batch_start_time = time.time()
                volume = torch.flatten(volume, start_dim=0, end_dim=1) # (b*3,2,80,80,80)
                target = torch.flatten(target, start_dim=0, end_dim=1) # (b*3,2,80,80,80)
                target_weight = torch.flatten(target_weight, start_dim=0, end_dim=1) # (b*3,2,1)
                edge_radius = torch.flatten(edge_radius, start_dim=0, end_dim=1) # (b*3,1)
                edge_volume = torch.flatten(edge_volume, start_dim=0, end_dim=1) # (b*3,1)

                volume = volume.type(torch.FloatTensor).cuda(non_blocking=True)
                target = target.type(torch.FloatTensor).cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                batch_size = volume.size(0)
                output= self.model(volume)

                metric = accuracy(output.detach().cpu().numpy(),
                                  target.detach().cpu().numpy(),
                                  target_weight.detach().cpu().numpy(),
                                  edge_radius.detach().cpu().numpy(),
                                  edge_volume.detach().cpu().numpy(),
                                  mode='test'
                                  )

                ap.update(metric['ap'], metric['val_batch_size'])
                ap_50.update(metric['ap_50'], metric['val_batch_size'])
                ap_75.update(metric['ap_75'], metric['val_batch_size'])
                ap_s.update(metric['ap_s'], metric['s_num'])
                ap_m.update(metric['ap_m'], metric['m_num'])
                ap_l.update(metric['ap_l'], metric['l_num'])
                ap_kp1.update(metric['ap_kp1'], metric['num_kp1'])
                ap_kp2.update(metric['ap_kp2'], metric['num_kp2'])
                exp_dis.update(metric['exp_dis'], metric['vis_keypoint'])
                exp_dis_kp1.update(metric['exp_dis_kp1'],metric['vis_kp1'])
                exp_dis_kp2.update(metric['exp_dis_kp2'],metric['vis_kp2'])


                # measure elapsed time
                batch_time.update(time.time() - batch_start_time)
                currentSteps += 1

                if currentSteps % 10 == 0:
                    print('Step:[{}/{}] | Time:{:.0f} | EXP_D:{:.4f} | AP:{:.3f} | EXP_D_KP1:{:.4f} | EXP_D_KP2:{:.4f} '.format(
                        self.currentSteps,self.totalSteps, 
                        batch_time.sum,
                        exp_dis.val,
                        ap.val,
                        exp_dis_kp1.val,
                        exp_dis_kp2.val
                        ))


            custom_print(Fore.GREEN + "Test Completed!", text_width=self.bar_len)
            print('Test Time:{:.0f}, AP:{:.3f}, AP_50:{:.3f}, AP_75:{:.3f}, AP_S:{:.3f}, AP_M:{:.3f}, AP_L:{:.3f}'.format(
                batch_time.sum,
                ap.avg, 
                ap_50.avg,
                ap_75.avg,
                ap_s.avg,
                ap_m.avg,
                ap_l.avg,
                ))
            print('AP_KP1:{:.3f}, AP_KP2:{:.3f}, EXP_D:{:.4f}, EXP_D_KP1:{:.4f}, EXP_D_KP2:{:.4f}'.format(
                ap_kp1.avg,
                ap_kp2.avg,
                exp_dis.avg,
                exp_dis_kp1.avg, 
                exp_dis_kp2.avg
                ))
        print("\n")

    def model_summary(self, input_size=None):
        if not input_size:
            input_size = (2, self.sub_size, self.sub_size, self.sub_size)

        custom_print(Fore.YELLOW + "Model Summary:" + self.model_name, text_width=self.bar_len)
        # summary(self.model, input_size=input_size, batch_size=1, device="cuda")
        print("*" * self.bar_len)
        print()

        flops, params = get_model_complexity_info(self.model, input_size, as_strings=True, print_per_layer_stat=False)
        custom_print('Computational complexity {}:{}'.format(self.model_name, flops), self.bar_len, '-')
        custom_print('Number of parameters (Enhace-Gen):{}'.format(params), self.bar_len, '-')

        config_shower(self.model_name)
        print("*" * self.bar_len)


    def model_load(self, ckpt_path):

        custom_print(Fore.RED + "Loading pretrained weight", text_width=self.bar_len)
        previous_weight = torch.load(ckpt_path)
        self.model.module.load_state_dict(previous_weight['stateDict'])
        self.optimizer.load_state_dict(previous_weight['optimizer'])
        self.currentSteps = int(previous_weight['step'])
        # print('current step: {}'.format(self.currentSteps))

        custom_print(Fore.YELLOW + "Weights loaded successfully", text_width=self.bar_len)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0