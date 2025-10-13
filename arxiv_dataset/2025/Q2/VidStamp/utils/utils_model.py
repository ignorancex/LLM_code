import torch
import copy
import os
from diffusers import DiffusionPipeline
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_name, args):
    print(f"Loading model: {model_name}")

    model = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    model.safety_checker = None
    original_vae = copy.deepcopy(model.vae)
    # Freeze parameters
    for param in original_vae.parameters():
        param.requires_grad = False

    # If needs loading from checkpoint
    if args.finetuning_stage == "second":
        if os.path.exists(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.vae.decoder.load_state_dict(checkpoint)
            print("Fine-tuned decoder loaded successfully.")
        else:
            print("Checkpoint not found. Using the default decoder.")
        
    
    # Ensure model has a VAE component
    if not hasattr(model, "vae") or not hasattr(model.vae, "decoder"):
        raise AttributeError(f"The model {model_name} does not contain a VAE decoder. Ensure it's a latent video model.")

    # Freeze everything except the decoder
    for param in model.vae.parameters():
        param.requires_grad = False  # Freeze the entire VAE
    for param in model.vae.decoder.parameters():
        param.requires_grad = True  # Fine-tune only the decoder

    return model.vae, original_vae

def load_msg_decoder(args, device="cuda"):
    print(f'>>> Building hidden decoder with weights from {args.msg_decoder_path}...')
    
    if 'torchscript' in args.msg_decoder_path:
        msg_decoder = torch.jit.load(args.msg_decoder_path).to(device)
    else:
        from utils_model import get_hidden_decoder, get_hidden_decoder_ckpt
        msg_decoder = get_hidden_decoder(num_bits=args.num_bits, redundancy=args.redundancy, num_blocks=args.decoder_depth, channels=args.decoder_channels).to(device)
        ckpt = get_hidden_decoder_ckpt(args.msg_decoder_path)
        print(msg_decoder.load_state_dict(ckpt, strict=False))
        msg_decoder.eval()

        print(f'>>> Whitening...')
        with torch.no_grad():
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            loader = get_dataloader(args.train_dir, transform, batch_size=16, num_imgs=16*500, collate_fn=None)
            ys = []
            for i, x in enumerate(loader):
                x = x.to(device)
                y = msg_decoder(x)
                ys.append(y.to('cpu'))
            ys = torch.cat(ys, dim=0)
            
            mean = ys.mean(dim=0, keepdim=True)
            ys_centered = ys - mean
            cov = ys_centered.T @ ys_centered
            e, v = torch.linalg.eigh(cov)
            L = torch.diag(1.0 / torch.pow(e, exponent=0.5))
            weight = torch.mm(L, v.T)
            bias = -torch.mm(mean, weight.T).squeeze(0)
            linear = nn.Linear(ys.shape[1], ys.shape[1], bias=True)
            linear.weight.data = np.sqrt(ys.shape[1]) * weight
            linear.bias.data = np.sqrt(ys.shape[1]) * bias
            msg_decoder = nn.Sequential(msg_decoder, linear.to(device))
            torchscript_m = torch.jit.script(msg_decoder)
            args.msg_decoder_path = args.msg_decoder_path.replace(".pth", "_whit.pth")
            print(f'>>> Creating torchscript at {args.msg_decoder_path}...')
            torch.jit.save(torchscript_m, args.msg_decoder_path)
    
    msg_decoder.eval()
    return msg_decoder