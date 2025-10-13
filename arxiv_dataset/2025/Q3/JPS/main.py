import argparse
import yaml
import logging
import os
from datetime import datetime
import sys
import io
import torch
import gc
sys.path.append('model/LLaVA_NeXT')

from dataset import DataLoaderFactory
from preprocess import PreprocessorFactory
from model import ModelFactory
from attack import AttackFactory
from revision import MultiAgents, SingleAgent
from judge import JudgeFactory

def setup_logger(config_file, log_dir):
    # 获取当前时间戳
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    os.makedirs(log_dir, exist_ok=True)
    path_from_config = config_file.split('config/', 1)[-1]
    base_filename = os.path.splitext(path_from_config.replace('/', '_'))[0]
    log_filename = os.path.join(log_dir, f"{timestamp}_{base_filename}.log")

    # 设置日志格式
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    # 配置日志处理器
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # 还可以同时输出到控制台
        ]
    )
    return logging.getLogger()


# 全局异常处理函数
def log_uncaught_exception(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        # 让系统默认处理 KeyboardInterrupt 异常
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    else:
        # 记录所有未捕获的异常
        logger = logging.getLogger()
        logger.error("未捕获的异常", exc_info=(exc_type, exc_value, exc_tb))

# 设置自定义异常处理器
sys.excepthook = log_uncaught_exception

class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)

def get_img_path(config, iteration_num):
    attack_type = config['attack_config']['attack_type']
    annotation = config['attack_config']['annotation']
    adv_save_dir = config['attack_config']['adv_save_dir']
    revision_iteration_num = config['attack_config']['revision_iteration_num']

    if attack_type not in ['text_only', 'white', 'related_image']:
        perturbation_constraint = config['attack_config']['perturbation_constraint']
        learning_rate = config['attack_config']['learning_rate']

    if attack_type == 'full':
        adv_img_file_name = f"full_con{str(perturbation_constraint).replace('/','|')}_lr{str(learning_rate).replace('/','|')}_revision{revision_iteration_num}_iteration{iteration_num}_{annotation}.png"
    elif attack_type == 'patch':
        adv_img_file_name = f"patch_lr{str(learning_rate).replace('/','|')}_revision{revision_iteration_num}_iteration{iteration_num}_{annotation}.png"
    elif attack_type == 'grid':
        adv_img_file_name = f"grid{config['attack_config']['grid_size']}_con{str(perturbation_constraint).replace('/','|')}_lr{str(learning_rate).replace('/','|')}_revision{revision_iteration_num}_iteration{iteration_num}_{annotation}.png"
    
    # only for providing name
    elif attack_type in ['text_only', 'white', 'related_image']:
        adv_img_file_name = f"{attack_type}_{annotation}.png"

    adv_img_path = os.path.join(adv_save_dir, config['inference_config']['attack_dataset'],config['model']['name'], adv_img_file_name)

    return adv_img_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/workshop/crm/project/patch_attack/config/mmsafetybench/internvl2/blank_con32|255_total0.yaml")
    parser.add_argument("--log_dir", type=str, default="process_logs")
    parser.add_argument("--run_type", type=str, default="attack")
    parser.add_argument("--multi_agent", type=int, default=1)
    parser.add_argument("--tqdm_log", type=bool, default=False)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger = setup_logger(args.config, args.log_dir)
    if args.tqdm_log is True:
        config['tqdm_out'] = TqdmToLogger(logger,level=logging.INFO)
    else:
        config['tqdm_out'] = None

    # 模型加载
    model = ModelFactory.get_model(config)
    logging.info(f"{config['model']['name']} loading finished")

    # 数据加载
    data_loader = DataLoaderFactory.get_loader(config["data_loader"])
    logging.info(f"{config['data_loader']['dataset_name']} loading finished")

    # 数据预处理
    preprocessor = PreprocessorFactory.get_preprocessor(config)
    data = preprocessor.preprocess(data_loader.data, model.pipeline)
    logging.info(f"{config['data_loader']['dataset_name']} processing finished")

    if args.run_type == 'attack':
        attack_prompt_list = [''] * len(config['attack_config']['multi_angle'])
        steer_prompt = ''

        attacker = AttackFactory.get_attack(config, model.pipeline)
        adv_img_path, attack_step = attacker.attack(data)

        logging.info(f"0th Image Attack finished, adv image saved at {adv_img_path}")
        logging.info(f"Attack using {attack_step} steps.")
        
        for i in range(config['attack_config']['revision_iteration_num']):
            responses, type_list = model.test(data, adv_img_path, steer_prompt, save = False)
            del model, attacker
            torch.cuda.empty_cache()
            gc.collect()
            
            if args.multi_agent:
                prompt_agents = MultiAgents(config)
            else:
                logging.info('Attention! Using Single Agent!')
                config['attack_config']['batch_size'] = 1
                config['inference_config']['batch_size'] = 10
                prompt_agents = SingleAgent(config)

            attack_prompt_list = prompt_agents.attack(data, responses, attack_prompt_list)
            del prompt_agents
            torch.cuda.empty_cache()
            gc.collect()
            steer_prompt = '\n'.join(attack_prompt_list) + '\n\n'

            logging.info(f"In the {i}-th iteration, steer prompt: {steer_prompt}")            
            
            model = ModelFactory.get_model(config)

            attacker = AttackFactory.get_attack(config, model.pipeline)
            adv_img_path, attack_step = attacker.attack(data, adv_img_path, steer_prompt, i+1)

            jb_prompt_txt_path = os.path.splitext(adv_img_path)[0] + '.txt'
            with open(jb_prompt_txt_path, 'w') as writer:
                writer.write(steer_prompt)

            logging.info(f"In the {i}-th iteration, Adv image saved at {adv_img_path}")
            logging.info(f"Attack using {attack_step} steps.")
    
        inference_result_path = model.test(data, adv_img_path, steer_prompt)
        del model, attacker
        torch.cuda.empty_cache()
        gc.collect()

        logging.info(f"{config['attack_config']['revision_iteration_num']}th Result saved at {inference_result_path}")

    elif args.run_type == 'inference':
        adv_img_path = get_img_path(config, config['attack_config']['revision_iteration_num'])
        jb_txt_path = os.path.splitext(adv_img_path)[0] + '.txt'
        logging.info(f"Attack image path: {adv_img_path}")

        steer_prompt = ''
        if os.path.exists(jb_txt_path):
            with open(jb_txt_path) as reader:
                for line in reader:
                    steer_prompt += line

        # 使用攻击的图片进行推理   
        inference_result_path = model.test(data, adv_img_path, steer_prompt)
        del model
        torch.cuda.empty_cache()
        gc.collect()
        logging.info(f"Inference finished, result saved at {inference_result_path}")
    else:
        raise ValueError("Invalid run_type, please choose from 'attack' or 'inference'.")

    # 评估攻击效果
    judge_model_list = config['judge_config']['model_list']
    for judge_model in judge_model_list:
        judger = JudgeFactory.get_judger(config, judge_model)
        judge_result_path, asr = judger.judge(inference_result_path)
        logging.info(f"Judge finished, result saved at {judge_result_path}")
        logging.info(f"{judge_result_path} ASR: {asr}")
        del judger
        torch.cuda.empty_cache()
        gc.collect()

    logging.info(f"The whole process has been completed, Thank you for using! :)")