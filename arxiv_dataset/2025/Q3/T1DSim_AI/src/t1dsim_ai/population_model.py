import torch
import torch.nn as nn


class WeightClipper(object):
    def __init__(self, min=-1, max=1):

        self.min = min
        self.max = max

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "weight"):
            w = module.weight.data
            w = w.clamp(self.min, self.max)


class CGMOHSUSimStateSpaceModel_V2(nn.Module):
    def __init__(
        self, n_feat=None, lookback_inputs=[None, None], scale_dx=1.0, init_small=True
    ):
        super(CGMOHSUSimStateSpaceModel_V2, self).__init__()
        self.n_feat = n_feat
        self.scale_dx = scale_dx

        self.input_insulin = 1
        self.input_carbs = 1
        # self.input_insulin = (
        #    1 if lookback_inputs[0] == None else int(lookback_inputs[0] * 12)
        # )
        # self.input_carbs = (
        #    1 if lookback_inputs[1] == None else int(lookback_inputs[1] * 12)
        # )

        # NN1(s1,u_I)
        self.net_dS1 = nn.Sequential(
            nn.Linear(1 + self.input_insulin, self.n_feat["S1"]),
            nn.ReLU(),
            nn.Linear(self.n_feat["S1"], 1),
        )

        # NN2(s1,s2)
        self.net_dS2 = nn.Sequential(
            nn.Linear(2, self.n_feat["S2"]),
            nn.ReLU(),
            nn.Linear(self.n_feat["S2"], 1),
        )

        # NN3(s2,I)
        self.net_dI = nn.Sequential(
            nn.Linear(2, self.n_feat["I"]),
            nn.ReLU(),
            nn.Linear(self.n_feat["I"], 1),
        )

        # NN4(I,x1)
        self.net_dX1 = nn.Sequential(
            nn.Linear(2, self.n_feat["X1"]),
            nn.ReLU(),
            nn.Linear(self.n_feat["X1"], 1),
        )

        # NN5(I,x2)
        self.net_dX2 = nn.Sequential(
            nn.Linear(2, self.n_feat["X2"]),
            nn.ReLU(),
            nn.Linear(self.n_feat["X2"], 1),
        )

        # NN6(I,x3)
        self.net_dX3 = nn.Sequential(
            nn.Linear(2, self.n_feat["X3"]),
            nn.ReLU(),
            nn.Linear(self.n_feat["X3"], 1),
        )

        # NN7(x1,x3,q1,q2,u_G)
        self.net_dQ1 = nn.Sequential(
            nn.Linear(5, self.n_feat["Q1"]),
            nn.ReLU(),
            nn.Linear(self.n_feat["Q1"], 1),
        )

        # NN8(x1,x2,q2,q1)
        self.net_dQ2 = nn.Sequential(
            nn.Linear(4, self.n_feat["Q2"]),
            nn.ReLU(),
            nn.Linear(self.n_feat["Q2"], 1),
        )

        # NN9(c1,c2)
        self.net_dC2 = nn.Sequential(
            nn.Linear(2, self.n_feat["C2"]),
            nn.ReLU(),
            nn.Linear(self.n_feat["C2"], 1),
        )

        # NN10(u_carbs,c1)
        self.net_dC1 = nn.Sequential(
            nn.Linear(1 + self.input_carbs, self.n_feat["C1"]),
            nn.ReLU(),
            nn.Linear(self.n_feat["C1"], 1),
        )

        clipper = WeightClipper()
        # clipper_pos = WeightClipper(0, 1)
        # print('Inside the models!')
        # Small initialization is better for multi-step methods
        if init_small:

            networks = {
                "S1": self.net_dS1,
                "S2": self.net_dS2,
                "I": self.net_dI,
                "X1": self.net_dX1,
                "X2": self.net_dX2,
                "X3": self.net_dX3,
                "Q1": self.net_dQ1,
                "Q2": self.net_dQ2,
                "C1": self.net_dC1,
                "C2": self.net_dC2,
            }

            # npa = 0
            for key in networks.keys():
                # print(key)
                net = networks[key]
                # for p in net.parameters():
                #    npa += p.numel()
                #    print(p.numel())
                for m in net.modules():

                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0, std=1e-4)
                        nn.init.constant_(m.bias, val=0)

                if key in []:  # ['Ug','Ug2']:
                    # net.apply(clipper_pos)
                    net.apply(clipper)

                else:
                    net.apply(clipper)
                    # net.bias = None

    def forward(self, in_x, in_u):

        q1, q2, s1, s2, Ins, x1, x2, x3, c2, c1 = (
            in_x[..., [0]],
            in_x[..., [1]],
            in_x[..., [2]],
            in_x[..., [3]],
            in_x[..., [4]],
            in_x[..., [5]],
            in_x[..., [6]],
            in_x[..., [7]],
            in_x[..., [8]],
            in_x[..., [9]],
        )

        u_I, u_carbs = in_u[..., [0]], in_u[..., 1:]

        # NN1(s1,u_I)
        in_1 = torch.cat((s1, u_I), -1)  # concatenate
        dS1 = self.net_dS1(in_1)

        # NN2(s1,s2)
        in_2 = torch.cat((s1, s2), -1)  # concatenate
        dS2 = self.net_dS2(in_2)

        # NN3(s2,I)
        in_3 = torch.cat((s2, Ins), -1)  # concatenate
        dI = self.net_dI(in_3)

        # NN4(I,x1)
        in_4 = torch.cat((Ins, x1), -1)  # concatenate
        dX1 = self.net_dX1(in_4)

        # NN5(I,x2)
        in_5 = torch.cat((Ins, x2), -1)  # concatenate
        dX2 = self.net_dX2(in_5)

        # NN6(I,x3)
        in_6 = torch.cat((Ins, x3), -1)  # concatenate
        dX3 = self.net_dX3(in_6)

        # NN7(x1,x3,q1,q2,u_G)
        in_7 = torch.cat((x1, x3, q1, q2, c2), -1)  # concatenate
        # print(in_7.requires_grad)
        dQ1 = self.net_dQ1(in_7)

        # NN8(x1,x2,q2,q1)
        in_8 = torch.cat((x1, x2, q1, q2), -1)  # concatenate
        dQ2 = self.net_dQ2(in_8)

        # NN9(c1,c2)
        in_9 = torch.cat((c1, c2), -1)  # concatenate
        dC2 = self.net_dC2(in_9)

        # NN10(u_carbs)
        in_10 = torch.cat((c1, u_carbs), -1)  # concatenate
        dC1 = self.net_dC1(in_10)

        # the state derivative is built by concatenation of the 9 nn, possibly scaled for numerical convenience
        dx = torch.cat((dQ1, dQ2, dS1, dS2, dI, dX1, dX2, dX3, dC2, dC1), -1)

        return dx
