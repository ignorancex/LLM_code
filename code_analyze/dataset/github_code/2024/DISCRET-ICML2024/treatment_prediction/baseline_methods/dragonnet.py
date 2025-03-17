import torch
import torch.nn as nn
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from transformers import CLIPModel, CLIPProcessor


def make_dragonet_backbone(input_size, hidden_size):
    return torch.nn.Sequential(nn.Linear(input_size, hidden_size), nn.ELU(), nn.Linear(hidden_size, hidden_size),
                          nn.ELU(), nn.Linear(hidden_size,hidden_size), nn.ELU(), nn.Linear(hidden_size, hidden_size))

def make_dragonet_backbone2(input_size, hidden_size):
    # return torch.nn.Sequential(nn.Linear(input_size, hidden_size), nn.ELU(), nn.Linear(hidden_size, hidden_size))
    return nn.Linear(input_size, hidden_size)

def make_dragonet_backbone_ls(input_size, hidden_size, num_class=None, cont_treatment=False):
    neural_net_ls = []
    for _ in range(input_size):
        neural_net_ls.append(nam_base(1, hidden_size, num_class=num_class, cont_treatment=cont_treatment))
    
    return torch.nn.ModuleList(neural_net_ls)





class dragonet_model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_treatment=2, num_class=None, cont_treatment=False):
        super(dragonet_model, self).__init__()
        self.backbone = make_dragonet_backbone(input_size, hidden_size)
        # self.t_predictions = torch.nn.Linear(hidden_size, 1)
        
        if num_class is None:
            self.y_layers   = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(hidden_size, int(hidden_size/2)), torch.nn.Linear(int(hidden_size/2), 1)) for _ in range(num_treatment)])
        else:
            self.y_layers   = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(hidden_size, int(hidden_size/2)), torch.nn.Linear(int(hidden_size/2), num_class)) for _ in range(num_treatment)])
        self.num_treatment = num_treatment
        # self.treatment_net = nn.Linear(hidden_size, 1)
        
        # for i in range(num_treatment):
            
        #     setattr(self, "y%s_hidden"%i, torch.nn.Linear(hidden_size, int(hidden_size/2)))
        #     setattr(self, "y%s_predictions"%i, torch.nn.Linear(int(hidden_size/2), 1))
        
        # self.y0_hidden = torch.nn.Linear(hidden_size, int(hidden_size/2))
        # self.y1_hidden = torch.nn.Linear(hidden_size, int(hidden_size/2))


        # self.y0_predictions = torch.nn.Linear(int(hidden_size/2), 1)
        # self.y1_predictions = torch.nn.Linear(int(hidden_size/2), 1)
    def encoding_features(self, x):
        hidden = self.backbone(x)
        return hidden

    def forward(self, x, a, d=None, test=False, backbone_out=None):
        if backbone_out is None:
            backbone_out = self.backbone(x)
        # t_pred = torch.sigmoid(self.t_predictions(backbone_out))
        # y_pred0 = self.y0_predictions(self.y0_hidden(backbone_out))
        # y_pred1 = self.y1_predictions(self.y1_hidden(backbone_out))
        y_pred_ls = []
        for k in range(self.num_treatment):
            y_pred = self.y_layers[k](backbone_out)
            y_pred_ls.append(y_pred)
            
        y_pred_all = torch.stack(y_pred_ls, dim=1)
        if not test:      
            
            return backbone_out, y_pred_all[torch.arange(len(y_pred_all)),a.view(-1).long()]
        else:
            
            return backbone_out, y_pred_all[torch.arange(len(y_pred_all)),a.view(-1).long()], y_pred_all

class nam_base(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_treatment=2, num_class=None, cont_treatment=False):
        super(nam_base, self).__init__()
        self.backbone = make_dragonet_backbone(input_size, hidden_size)            
        # self.t_predictions = torch.nn.Linear(hidden_size, 1)
        self.cont_treatment= cont_treatment
        if not cont_treatment:
            if num_class is None:
                self.y_layers   = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(hidden_size, int(hidden_size/2)), torch.nn.Linear(int(hidden_size/2), 1)) for _ in range(num_treatment)])
            else:
                self.y_layers   = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(hidden_size, int(hidden_size/2)), torch.nn.Linear(int(hidden_size/2), num_class)) for _ in range(num_treatment)])
            self.num_treatment = num_treatment
        else:
            self.y_layers   = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(hidden_size, int(hidden_size/2)), torch.nn.Linear(int(hidden_size/2), 1)) for _ in range(1)])    
            self.num_treatment = 1
        
        # self.treatment_net = nn.Linear(hidden_size, 1)
        
        # for i in range(num_treatment):
            
        #     setattr(self, "y%s_hidden"%i, torch.nn.Linear(hidden_size, int(hidden_size/2)))
        #     setattr(self, "y%s_predictions"%i, torch.nn.Linear(int(hidden_size/2), 1))
        
        # self.y0_hidden = torch.nn.Linear(hidden_size, int(hidden_size/2))
        # self.y1_hidden = torch.nn.Linear(hidden_size, int(hidden_size/2))


        # self.y0_predictions = torch.nn.Linear(int(hidden_size/2), 1)
        # self.y1_predictions = torch.nn.Linear(int(hidden_size/2), 1)
        
    def forward(self, x, a, d=None, test=False):
        backbone_out = self.backbone(x)
        # t_pred = torch.sigmoid(self.t_predictions(backbone_out))
        # y_pred0 = self.y0_predictions(self.y0_hidden(backbone_out))
        # y_pred1 = self.y1_predictions(self.y1_hidden(backbone_out))
        y_pred_ls = []
        for k in range(self.num_treatment):
            y_pred = self.y_layers[k](backbone_out)
            y_pred_ls.append(y_pred)
            
        y_pred_all = torch.stack(y_pred_ls, dim=1)
        if self.cont_treatment:
            return backbone_out, y_pred_ls[0]
        else:
            if not test:      
                
                return backbone_out, y_pred_all[torch.arange(len(y_pred_all)),a.view(-1).long()]
            else:
                
                return backbone_out, y_pred_all[torch.arange(len(y_pred_all)),a.view(-1).long()], y_pred_all
        

class dragonet_model_image(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_treatment=2):
        super(dragonet_model_image, self).__init__()
        self.img_emb = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.img_emb.visual_projection = nn.Linear(self.img_emb.visual_projection.in_features, hidden_size)
        
        self.backbone = make_dragonet_backbone(input_size, hidden_size)
        # self.t_predictions = torch.nn.Linear(hidden_size, 1)
        
        self.y_layers   = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(hidden_size*2, int(hidden_size/2)), torch.nn.Linear(int(hidden_size/2), 1)) for _ in range(num_treatment)])
        self.num_treatment = num_treatment
        # self.treatment_net = nn.Linear(hidden_size, 1)
        
        # for i in range(num_treatment):
            
        #     setattr(self, "y%s_hidden"%i, torch.nn.Linear(hidden_size, int(hidden_size/2)))
        #     setattr(self, "y%s_predictions"%i, torch.nn.Linear(int(hidden_size/2), 1))
        
        # self.y0_hidden = torch.nn.Linear(hidden_size, int(hidden_size/2))
        # self.y1_hidden = torch.nn.Linear(hidden_size, int(hidden_size/2))


        # self.y0_predictions = torch.nn.Linear(int(hidden_size/2), 1)
        # self.y1_predictions = torch.nn.Linear(int(hidden_size/2), 1)
        
    def forward(self, images, x, a, d=None, test=False):
        
        image_emb = self.img_emb.get_image_features(**images)
        
        # hidden = self.hidden_features(x)
        hidden = self.backbone(x)
        t_hidden = hidden
        t_hidden = torch.cat([t_hidden, image_emb], dim=1)
        
        
        # t_pred = torch.sigmoid(self.t_predictions(backbone_out))
        # y_pred0 = self.y0_predictions(self.y0_hidden(backbone_out))
        # y_pred1 = self.y1_predictions(self.y1_hidden(backbone_out))
        y_pred_ls = []
        for k in range(self.num_treatment):
            y_pred = self.y_layers[k](t_hidden)
            y_pred_ls.append(y_pred)
            
        y_pred_all = torch.cat(y_pred_ls, dim=-1)
        if not test:      
            
            return y_pred_all[torch.arange(len(y_pred_all)),a.view(-1).long()]
        else:
            
            return y_pred_all[torch.arange(len(y_pred_all)),a.view(-1).long()], y_pred_all
  
        
class dragonet_model_nam(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_treatment=2, cont_treatment=False, has_dose=False, num_class=None):
        super(dragonet_model_nam, self).__init__()
        if cont_treatment:
            num_treatment=1

        self.backbone_ls = make_dragonet_backbone_ls(input_size, hidden_size, num_class=num_class, cont_treatment=cont_treatment)
        # if cont_treatment:
        #     self.treatment_net = nam_base(1, hidden_size, num_class=num_class)
        if has_dose:
            self.dose_net = nam_base(1, hidden_size, num_class=num_class)
        self.has_dose = has_dose
        # self.backbone = make_dragonet_backbone(input_size, hidden_size)
        # self.t_predictions = torch.nn.Linear(hidden_size, 1)
        
        # self.y_layers   = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(hidden_size, int(hidden_size/2)), torch.nn.Linear(int(hidden_size/2), 1)) for _ in range(num_treatment)])
        self.num_treatment = num_treatment
        self.cont_treatment = cont_treatment
        # self.treatment_net = nn.Linear(hidden_size, 1)
        
        # for i in range(num_treatment):
            
        #     setattr(self, "y%s_hidden"%i, torch.nn.Linear(hidden_size, int(hidden_size/2)))
        #     setattr(self, "y%s_predictions"%i, torch.nn.Linear(int(hidden_size/2), 1))
        
        # self.y0_hidden = torch.nn.Linear(hidden_size, int(hidden_size/2))
        # self.y1_hidden = torch.nn.Linear(hidden_size, int(hidden_size/2))


        # self.y0_predictions = torch.nn.Linear(int(hidden_size/2), 1)
        # self.y1_predictions = torch.nn.Linear(int(hidden_size/2), 1)
    
    def encoding_features(self, x):
        if not self.cont_treatment and self.has_dose:
            hidden = self.feature_weight(self.linear1(x))
        else:
            hidden = self.feature_weight(x)
        return hidden

    def forward(self, x, a, d=None, test=False):
        sum_out = 0
        full_sum_out = 0
        backbone_sum_out = 0
        
        for i in range(len(self.backbone_ls)):
            if test and not self.cont_treatment:
                backbone_out, out, full_out = self.backbone_ls[i](x[:,i].view(-1,1), a, d=d, test=test)
                backbone_sum_out += backbone_out
                sum_out += out
                full_sum_out += full_out
            else:
                backbone_out, out = self.backbone_ls[i](x[:,i].view(-1,1), a, d=d, test=test)
                backbone_sum_out += backbone_out
                sum_out += out        
        if self.has_dose:
            if test and not self.cont_treatment:
                backbone_out, out, full_out = self.dose_net(d.view(-1,1), a, d=d, test=test)
                full_sum_out += full_out
            else:
                backbone_out, out = self.dose_net(d.view(-1,1), a, d=d, test=test)
            backbone_sum_out += backbone_out
            sum_out += out
        # backbone_out = self.backbone(x)
        # # t_pred = torch.sigmoid(self.t_predictions(backbone_out))
        # # y_pred0 = self.y0_predictions(self.y0_hidden(backbone_out))
        # # y_pred1 = self.y1_predictions(self.y1_hidden(backbone_out))
        # y_pred_ls = []
        # for k in range(self.num_treatment):
        #     y_pred = self.y_layers[k](backbone_out)
        #     y_pred_ls.append(y_pred)
            
        # y_pred_all = torch.cat(y_pred_ls, dim=-1)
        if not test or self.cont_treatment:      
            
            return backbone_sum_out, sum_out
        else:
            
            return backbone_sum_out, sum_out, full_sum_out
        
    def get_topk_features(self, x, t, d=None, k=2, test=True):
        
        coeff_ls = []
        for i in range(len(self.backbone_ls)):
            if test and not self.cont_treatment:
                backbone_out, out, full_out = self.backbone_ls[i](x[:,i].view(-1,1), t, d=d, test=test)
                coeff = out.view(-1)/(x[:,i].view(-1) + 1e-4)
            else:
                backbone_out, out = self.backbone_ls[i](x[:,i].view(-1,1), t, d=d, test=test)
                coeff = out.view(-1)/(x[:,i].view(-1) + 1e-4)
            coeff_ls.append(coeff)
        coeff_tensor = torch.stack(coeff_ls, dim=-1)
        return torch.topk(torch.abs(coeff_tensor), k, dim=-1)[1]
                
            