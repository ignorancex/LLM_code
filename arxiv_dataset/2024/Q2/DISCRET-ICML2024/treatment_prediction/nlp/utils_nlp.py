import torch
def load_pretrained_backbone_models_rl(mod, model_path):
    cached_model = torch.load(model_path)
    
    mod.dqn.policy_net.backbone_model.distilbert.load_state_dict(cached_model.distilbert.state_dict())
    mod.dqn.target_net.backbone_model.distilbert.load_state_dict(cached_model.distilbert.state_dict())


def load_pretrained_backbone_models(mod, model_path):
    cached_model = torch.load(model_path)

    mod.model.distilbert.load_state_dict(cached_model.distilbert.state_dict())