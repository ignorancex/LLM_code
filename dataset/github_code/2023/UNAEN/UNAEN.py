import torch.nn as nn
import torch
import torch.optim as optim
from models import networks
import utils
import numpy as np
import os


class UNAEN(object):

    def __init__(self, config):

        self.n_disc = 1  # number of iteration for discriminator update in one epoch
        self.n_gen = 2  # number of iteration for generator update in one epoch

        # loss
        self.lambda_ssim = 0.5  # SSIM loss weight
        self.lambda_adv = 0.1  # adversarial loss weight
        self.lambda_cyc = 1  # cycle loss weight
        self.lambda_sty = 0.1  # style loss weight

        # network
        self.gen_A2B = networks.generator_RCAN(num_in_ch=1, num_out_ch=1, num_feat=64).to(config.device)
        self.gen_B2A = networks.generator_RCAN(num_in_ch=1, num_out_ch=1, num_feat=64).to(config.device)
        self.disc_A = networks.DiscriminatorVGG(in_ch=1).to(config.device)
        self.disc_B = networks.DiscriminatorVGG(in_ch=1).to(config.device)

        # Loss criterion
        self.loss_L1 = nn.L1Loss().to(config.device)
        self.loss_MSE = nn.MSELoss().to(config.device)
        self.loss_ssim = utils.SSIM_loss().to(config.device)

        # optimizer
        params_G = list(self.gen_A2B.parameters()) + list(self.gen_B2A.parameters())
        params_D = list(self.disc_A.parameters()) + list(self.disc_B.parameters())
        self.optimizer_G = optim.Adam(params_G, lr=1e-4, betas=(0.9, 0.99), amsgrad=True)
        self.optimizer_D = optim.Adam(params_D, lr=1e-4, betas=(0.9, 0.99), amsgrad=True)

        # Scheduler
        # self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=config.epochs, eta_min=1e-8, last_epoch=-1)
        # self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=config.epochs, eta_min=1e-8, last_epoch=-1)
        self.scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=10, gamma=0.5)
        self.scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=10, gamma=0.5)

        self.config = config
        self.start_epoch = 0
        # load weight
        if config.load_weight:
            self.load_weight()


    def load_weight(self):

        path = os.path.join(self.config.snap_path, 'best_ssim_network_parameter.pth')
        checkpoint = torch.load(path, map_location=torch.device(self.config.device))
        self.gen_A2B.load_state_dict(checkpoint['gen_A2B'])
        self.gen_B2A.load_state_dict(checkpoint['gen_B2A'])
        self.disc_A.load_state_dict(checkpoint['disc_A'])
        self.disc_B.load_state_dict(checkpoint['disc_B'])
        self.start_epoch = checkpoint['epoch']


    def train(self, data_loader):

        MA_loader = data_loader[0]             # MA: motion artifact
        MF_loader = data_loader[1]             # MF: motion free
        val_loader = data_loader[2]

        log_file = open(os.path.join(self.config.snap_path, 'log.txt'), "a+")
        max_ssim = 0.0

        for epoch in range(self.start_epoch, self.config.epochs):

            loss_D, loss_G = 0., 0.
            # generator
            self.gen_A2B.train()
            self.gen_B2A.train()
            # discriminator
            self.disc_A.train()
            self.disc_B.train()

            for step, ((A), (B)) in enumerate(zip(MA_loader, MF_loader)):

                ########################
                #       data load      #
                ########################
                MA_img = A['ma_img'].to(self.config.device)
                MF_img = B['gt_img'].to(self.config.device)

                real_label = torch.full((self.config.batch_size, 1), 1, dtype=MA_img.dtype).to(self.config.device)
                fake_label = torch.full((self.config.batch_size, 1), 0, dtype=MA_img.dtype).to(self.config.device)
                ###########################
                # (1) Update Disc network #
                ###########################
                self.optimizer_D.zero_grad()
                for i in range(self.n_disc):
                    # generator output
                    gen_MF = MA_img - self.gen_A2B(MA_img)
                    cyclic_MA = self.gen_B2A(gen_MF)

                    # output of discriminator
                    disc_MA = self.disc_A(MA_img)
                    disc_cyclic_MA = self.disc_A(cyclic_MA.detach())
                    disc_MF = self.disc_B(MF_img)
                    disc_gen_MF = self.disc_B(gen_MF.detach())

                    # reconstruction loss
                    loss_disc_MA_cycle = (self.loss_MSE(disc_cyclic_MA, fake_label) + self.loss_MSE(disc_MA, real_label)) / 2
                    # style loss
                    loss_disc_MF_sty = (self.loss_MSE(disc_gen_MF, fake_label) + self.loss_MSE(disc_MF, real_label)) / 2

                    # discriminator weight update
                    loss_D_total = loss_disc_MA_cycle + loss_disc_MF_sty
                    loss_D_total.backward()
                    self.optimizer_D.step()

                    loss_D += loss_D_total.item() / self.n_disc
                ##########################
                # (2) Update Gen network #
                ##########################
                self.optimizer_G.zero_grad()
                for i in range(self.n_gen):
                    # generator output
                    gen_MF = MA_img - self.gen_A2B(MA_img)
                    cyclic_MA = self.gen_B2A(gen_MF)

                    # output of discriminator
                    disc_cyclic_MA = self.disc_A(cyclic_MA)
                    disc_gen_MF = self.disc_B(gen_MF)

                    # L1 loss
                    loss_L1_MA = self.loss_L1(MA_img, cyclic_MA)
                    # SSIM loss
                    loss_ssim_MA = (1 - self.loss_ssim(MA_img, cyclic_MA) ** 2)
                    # adversatial loss
                    loss_adv_MA = self.loss_MSE(disc_cyclic_MA, real_label)
                    # reconstruction loss
                    loss_gen_MA_cycle = loss_L1_MA + self.lambda_ssim * loss_ssim_MA + self.lambda_adv * loss_adv_MA
                    # generator style loss
                    loss_gen_MF_sty = self.loss_MSE(disc_gen_MF, real_label)

                    # generator weight update
                    loss_G_total = self.lambda_cyc * loss_gen_MA_cycle + self.lambda_sty * loss_gen_MF_sty
                    loss_G_total.backward()
                    self.optimizer_G.step()

                    loss_G += loss_G_total.item() / self.n_gen
                ########################
                #     record loss      #
                ########################
                if (step + 1) % self.config.log_step == 0:
                    log_file.write("Epoch [{}/{}] Step [{}/{}] lr [{:.8f}]: loss_D_total={:.5f}  loss_G_total={:.5f}\n"
                          .format(epoch + 1,
                                  self.config.epochs,
                                  step + 1,
                                  len(MA_loader),
                                  self.optimizer_G.param_groups[0]['lr'],
                                  loss_D / self.config.log_step,
                                  loss_G / self.config.log_step))
                    print("Epoch [{}/{}] Step [{}/{}] lr [{:.8f}]: loss_D_total={:.5f}  loss_G_total={:.5f}"
                          .format(epoch + 1,
                                  self.config.epochs,
                                  step + 1,
                                  len(MA_loader),
                                  self.optimizer_G.param_groups[0]['lr'],
                                  loss_D / self.config.log_step,
                                  loss_G / self.config.log_step))
                    loss_D, loss_G = 0., 0.

            self.scheduler_D.step()
            self.scheduler_G.step()

            ########################
            #     validation       #
            ########################
            self.gen_A2B.eval()
            with torch.no_grad():
                print('Validation:')
                ssim_eval, psnr_eval = 0, 0
                for i, batch in enumerate(val_loader):
                    MA_img = batch['ma_img'].to(self.config.device)
                    MF_img = batch['gt_img'].to(self.config.device)
                    gen_MF = MA_img - self.gen_A2B(MA_img)
                    ssim_eval += utils.ssim(gen_MF, MF_img).item()
                    psnr_eval += utils.psnr(gen_MF, MF_img).item()
                ssim_eval = ssim_eval / len(val_loader)
                psnr_eval = psnr_eval / len(val_loader)
                print("Avg SSIM = {}, Avg PSNR = {}\n".format(ssim_eval, psnr_eval))
                log_file.write("Avg SSIM = {}, Avg PSNR = {}\n".format(ssim_eval, psnr_eval))

                if epoch == 0 or max_ssim < ssim_eval:
                    max_ssim = ssim_eval
                    weights_file = os.path.join(self.config.snap_path, 'best_ssim_network_parameter.pth')
                    torch.save({
                        'epoch': epoch,
                        'gen_A2B': self.gen_A2B.state_dict(),
                        'gen_B2A': self.gen_B2A.state_dict(),
                        'disc_A': self.disc_A.state_dict(),
                        'disc_B': self.disc_B.state_dict(),
                        }, weights_file)
                    print('save weights of epoch %d' % (epoch + 1) + '\n')
                    log_file.write('save weights of epoch %d' % (epoch + 1) + '\n')

            print("\n")
            log_file.write("\n")
        log_file.close()


    def eval(self, test_loader):

        self.load_weight()
        self.gen_A2B.eval()
        if not os.path.exists(self.config.inference_path):
            os.makedirs(self.config.inference_path)
        log_file = open(os.path.join(self.config.snap_path, 'log.txt'), "a+")
        with torch.no_grad():
            ma_ssim, ma_psnr = 0., 0.
            eval_ssim, eval_psnr = 0, 0
            for i, batch in enumerate(test_loader):

                MA_img = batch['ma_img'].to(self.config.device)
                MF_img = batch['gt_img'].to(self.config.device)

                extracted = self.gen_A2B(MA_img)
                gen_MF = MA_img - extracted

                ma_ssim += utils.ssim(MA_img, MF_img).item()
                ma_psnr += utils.psnr(MA_img, MF_img).item()
                eval_ssim += utils.ssim(gen_MF, MF_img).item()
                eval_psnr += utils.psnr(gen_MF, MF_img).item()

                gen_MF = gen_MF[0, 0].cpu().numpy()
                extracted = extracted[0, 0].cpu().numpy()
                np.savez(os.path.join(self.config.inference_path, batch['filename'][0]), pred=gen_MF, extracted=extracted)

        num = len(test_loader)

        print('\nBefore Reduction: ssim=%.4f and psnr=%.4f' % (ma_ssim / num, ma_psnr / num))
        log_file.write('\nBefore Reduction: ssim=%.4f and psnr=%.4f\n' % (ma_ssim / num, ma_psnr / num))

        print('\nEvaluation: ssim=%.4f and psnr=%.4f' % (eval_ssim / num, eval_psnr / num))
        log_file.write('\nEvaluation: ssim=%.4f and psnr=%.4f\n' % (eval_ssim / num, eval_psnr / num))

        log_file.close()