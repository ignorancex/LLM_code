import click
import hydra
import torch
from pathlib import Path
from omegaconf import OmegaConf
from icon.policies.base_policy import BasePolicy
from icon.env_runner import EnvRunner

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command(help="Evaluate policies in simulation.")
@click.option("-t", "--task", type=str, required=True, help="Task name.")
@click.option("-a", "--algo", type=str, required=True, help="Algorithm name.")
@click.option("-c", "--checkpoint", type=str, default="", help="Pretrained checkpoint.")
@click.option("-d", "--device", type=str, default="cuda", help="Device type.")
@click.option("-ne", "--num_episodes", type=int, default=50, help="Number of episodes.")
@click.option("-rm", "--render_mode", type=str, default="rgb_array", help="Rendering mode.")
def main(task, algo, checkpoint, device, num_episodes, render_mode):
    with hydra.initialize_config_dir(
        config_dir=str(Path(__file__).parent.parent.joinpath("icon/configs")),
        version_base="1.2" 
    ):
        overrides = [
            f'task={task}',
            f'algo={algo}',
            f'task.env_runner.num_episodes={num_episodes}',
            f'task.env_runner.env.render_mode={render_mode}',
        ]
        cfg = hydra.compose(config_name="config", overrides=overrides)
        env_runner: EnvRunner = hydra.utils.instantiate(cfg.task.env_runner)
        device = torch.device(device)
        policy: BasePolicy = hydra.utils.instantiate(cfg.algo.policy)
        policy.to(device)
        policy.eval()

        if any(checkpoint):
            state_dicts = torch.load(checkpoint, map_location=device)
            policy.load_state_dicts(state_dicts)
        else:
            checkpoint = f"checkpoints/{task}/{algo}.pth"
            state_dicts = torch.load(checkpoint, map_location=device)
            policy.load_state_dicts(state_dicts)

        env_runner.run(policy, device)
    

if __name__ == "__main__":
    main()