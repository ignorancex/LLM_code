import torch
import ops
from torchvision.utils import save_image

N_EPOCHS_LOCAL_CLF_TRAINING = 10  # this is valid for only Olivetti
N_EPOCHS_SAVE = 2

class Participant:

    def __init__(self,
                 id,
                 args,
                 train_loader,
                 clf,
                 clf_crit,
                 clf_optim,
                 c_attack=-1,
                 c_fake=-1,
                 ):
        self.id = id
        self.args = args
        self.train_loader = train_loader
        self.clf = clf
        self.clf_crit = clf_crit
        self.clf_optim = clf_optim
        self.c_attack = c_attack
        self.c_fake = c_fake

        global N_EPOCHS_LOCAL_CLF_TRAINING
        if self.args.dataset == "mnist":
            N_EPOCHS_LOCAL_CLF_TRAINING = 1

        # the participant is an attacker
        self.is_attacker = False
        if c_attack >= 0:
            self.is_attacker = True

            if args.dataset == "mnist":
                from networks.gen_mnist import Generator
                self.gen = Generator().to(args.device)
                self.gen.apply(ops.weights_init)
                self.gen_optim = torch.optim.SGD(self.gen.parameters(), 0.001, momentum=0.0)

            elif args.dataset == "olivetti":
                from networks.gen_olivetti import Generator
                self.gen = Generator().to(args.device)
                self.gen.apply(ops.weights_init)
                self.gen_optim = torch.optim.RMSprop(self.gen.parameters(), 0.0001, eps=1e-5)

            print("generator network of the attacker:", flush=True)
            print(self.gen, flush=True)
            print("optimizer for the generator:", flush=True)
            print(self.gen_optim, flush=True)

            # keep a copy of the original dataset
            self.data_orig = self.train_loader.dataset.data.clone().detach()
            self.targets_orig = self.train_loader.dataset.targets.clone().detach()

            # list to keep all generated images
            self.data_gen = []
            self.targets_gen = []


    def train_epoch(self, epoch):
        """
        If the participant is victim (honest):
            - trains its local generator
            - samples from the generator, and merges the samples with its local
            dataset
        All participants:
            - trains the classification model for 1 epoch.
        """

        # if the participant is an attacker,
        if self.is_attacker:
            # train the generator network
            print("training the local generator", flush=True)
            ops.train_gen(self.args, self.clf, self.gen, self.c_attack, self.clf_crit, self.gen_optim, self.args.n_steps_train_gen)

            # draw samples from the generator network and
            fake_images = ops.draw_samples_from(self.gen, self.args.device, self.args.n_fake_to_trainset, rescale=True)
            fake_labels = torch.zeros(fake_images.shape[0], dtype=torch.long)
            fake_labels.fill_(self.c_fake)

            self.data_gen.append(fake_images)
            self.targets_gen.append(fake_labels)

            # add the most recently-generated fake samples to the dataset
            data_new = [self.data_orig.unsqueeze(1), self.data_gen[-1]]
            targets_new = [self.targets_orig, self.targets_gen[-1]]
            for hix in range(2, self.args.n_previous_to_trainset + 1):
                if len(self.data_gen) >= hix:
                    data_new.append(self.data_gen[-hix])
                    targets_new.append(self.targets_gen[-hix])
            self.train_loader.dataset.data = torch.cat(data_new).squeeze(1)
            self.train_loader.dataset.targets = torch.cat(targets_new)
            print("data.shape:{}, targets.shape:{}".format(self.train_loader.dataset.data.shape, self.train_loader.dataset.targets.shape), flush=True)
            print("old vs new training set sizes: {} - {}".format(len(self.data_orig), len(self.train_loader.dataset)), flush=True)

            # save the generated images for visualization
            if epoch % N_EPOCHS_SAVE == 0:
                save_image(
                    fake_images[:100] / 255.0,
                    "{}/fake_part-{}_{:03d}.png".format(self.args.output_dir, self.id, epoch),
                    10,
                    2,
                    pad_value=255)

                # save the generator network
                torch.save(
                    {"gen": self.gen.state_dict(), "gen_optim": self.gen_optim.state_dict(), "epoch": epoch},
                    "{}/generator_part-{}_{:04d}.pth".format(self.args.output_dir, self.id, epoch)
                )

        # train the shared model for 1 epoch
        print("training the shared classifier for {} epochs".format(N_EPOCHS_LOCAL_CLF_TRAINING), flush=True)
        ops.train_clf(self.args, self.clf, self.train_loader, self.clf_crit, self.clf_optim, N_EPOCHS_LOCAL_CLF_TRAINING)

