"""
This script is designed to perform a prediction of stock price movements in relation to earnings surprises
over a given period. It integrates functionalities such as data fetching, model initialization and API interaction 
to facilitate a prediction of stock trends over a given tick and period.
"""

import requests
import json
import time
import argparse
import os
import sys
import io
import re
import warnings
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.dates as mdates
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from functools import partial
from transformers import AutoConfig, GenerationConfig, PretrainedConfig
from transformers import BitsAndBytesConfig
from transformers import pipeline, Pipeline

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MobileVLM'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MGM'))

from detect_anomaly import detect_recency_bias, detect_authority_bias
from data.fetch_sp500 import tickers_sp500, fetch_and_save_prices
from utils import inference_func, load_pretrained_func, get_model_name, set_all_seeds


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="llava", help='Model name')
    parser.add_argument('--api', action='store_true', default=False, help='Use API')
    parser.add_argument('--token', type=str, default="", help='API token')
    parser.add_argument('--prompt_cfg', type=str, default="config/prompt.json", help='Prompt JSON file')
    parser.add_argument('--model_cfg', type=str, default="config/model.json", help='Model JSON file')
    parser.add_argument('--celebrity_cfg', type=str, default="config/celebrity.json", help='celebrity JSON file')
    parser.add_argument('--stock_file', type=str, default="data/stock_history.csv", help='history stock data file path')
    parser.add_argument('--eps_dir', type=str, default="data/eps_history/", help='history eps data dir path')
    parser.add_argument('--ticker', type=str, default="AAPL", help='stock ticker')
    parser.add_argument('--start_time', type=str, default="2020-12-01", help='start time')
    parser.add_argument('--end_time', type=str, default="2021-01-01", help='end time')
    parser.add_argument('--narrative', action='store_true', default=False, help='use narrative input string')
    parser.add_argument('--image', action='store_true', default=False, help='use image input string')
    parser.add_argument('--load_8bit', action='store_true', default=False, help='use 8bit quantization')
    parser.add_argument('--load_4bit', action='store_true', default=False, help='use 4bit quantization')
    parser.add_argument('--bias_type', type=str, default="recency", help='bias type')
    parser.add_argument('--save', action='store_true', default=False, help='save output')
    parser.add_argument('--output_dir', type=str, default="output", help='Output directory')
    parser.add_argument('--batch_inference', action='store_true', default=False, help='use batch inference')
    
    return parser.parse_args()


def load_json(prompt_path, model_path, celebrity_path=None):
    with open(prompt_path, "r") as prompt_file:
        prompt_dict = json.load(prompt_file)

    with open(model_path, "r") as model_file:
        model_dict = json.load(model_file)
    
    celebrity_dict = None
    if celebrity_path:
        with open(celebrity_path, "r") as celebrity_file:
            celebrity_dict = json.load(celebrity_file)
        
    return prompt_dict, model_dict, celebrity_dict


def load_stock_data(path):
    prices = pd.read_csv(path, header=[0, 1], index_col=0)
    return prices


class client(object):
    def __init__(self, url=None, headers=None, openai=False, model=None, pipe=None, gen_cfg=None, image_input=False, batch_inference=False):
        self.url = url
        self.headers = headers
        self.openai = openai
        self.client = OpenAI() if openai else None
        self.model = model
        self.pipe = pipe
        self.gen_cfg = gen_cfg
        self.image_input = image_input
        self.batch_inference = batch_inference
    
    def query(self, batched_message, batched_image=None):
        if self.openai:
            if batched_image is not None:
                raise NotImplementedError("Image input is not implemented for OpenAI API")
            
            batched_response = []
            for message in batched_message:
                retry = 3
                while retry > 0:
                    try:
                        completion = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": message}],
                        )
                    except:
                        retry -= 1
                        completion = None
                        time.sleep(10)
                
                batched_response.append(completion.choices[0].message.content) if completion else batched_response.append("Missing")
            
            return batched_response
        
        elif not self.pipe:
            if batched_image is not None:
                raise NotImplementedError("Vision2Seq task is not implemented for Hugging Face Inference API")
            
            batched_response = []
            for message in batched_message:
                retry = 3
                while retry > 0:
                    try:
                        payload = {"inputs": message}
                        response = requests.post(self.url, headers=self.headers, json=payload)
                    except:
                        retry -= 1
                        response = None
                        time.sleep(10)
                    
                batched_response.append(response.json()[0]['generated_text']) if response else batched_response.append("Missing")
            
            return batched_response
        
        else:
            if self.batch_inference:
                if self.image_input:
                    batched_response = self.pipe(batched_message, batched_image)
                else:
                    batched_response = self.pipe(batched_message)
            else:
                batched_response = []
                if self.image_input:
                    for message, image in zip(batched_message, batched_image):
                        if isinstance(self.pipe, Pipeline):
                            batched_response.append(self.pipe(image, prompt=message, generate_kwargs=self.gen_cfg.to_dict())[0]['generated_text'])
                        else:
                            batched_response.append(self.pipe(message, image))
                else:
                    for message in batched_message:
                        if isinstance(self.pipe, Pipeline):
                            batched_response.append(self.pipe(message, generate_kwargs=self.gen_cfg.to_dict())[0]['generated_text'])
                        else:
                            batched_response.append(self.pipe(message))
            
            return batched_response
 
            
def init_model(model_dict, args):
    if args.api:
        if args.model in model_dict:
            if 'API_URL' in model_dict[args.model]:
                url = model_dict[args.model]['API_URL']
                headers = model_dict[args.model]['headers']
                headers['Authorization'] = f"Bearer {args.token}"
                return client(url=url, headers=headers, openai=False)
            else:
                raise ValueError(f"'API_URL' not found in model.json for model {args.model}")
        elif 'gpt' in args.model:
            model = model_dict[args.model]['model_id']
            return client(model=model, openai=True)
        else:
            raise ValueError(f"Model {args.model} not found in model.json")
    else:
        try:
            model_id = model_dict[args.model]['model_id']
            try:
                cfg = AutoConfig.from_pretrained(model_id)
                gen_cfg = GenerationConfig.from_pretrained(model_id)
                if gen_cfg.max_length == 20:
                    gen_cfg.max_length = 4096
                gen_cfg.pad_token_id = gen_cfg.pad_token_id if hasattr(gen_cfg, "pad_token_id") and gen_cfg.pad_token_id else cfg.pad_token_id if cfg and hasattr(cfg, "pad_token_id") and cfg.pad_token_id else 0
            except:
                cfg = PretrainedConfig(torch_dtype=torch.float16)
                gen_cfg = GenerationConfig(max_new_tokens=512, temperature=0.2)
            if args.image:
                load_pretrained_model = load_pretrained_func[args.model]
                if 'MobileVLM' == args.model:
                    tokenizer, model, image_processor, _ = load_pretrained_model(model_id, args.load_8bit, args.load_4bit)
                    conv_mode = "v1"
                    if args.batch_inference:
                        assert inference_func['MobileVLM']['batch'] is not None, f"batch inference function is not implemented for model {args.model}({model_id})"
                        pipe = partial(inference_func['MobileVLM']['batch'], model=model, tokenizer=tokenizer, image_processor=image_processor, conv_mode=conv_mode, generation_config=gen_cfg)
                    elif inference_func['MobileVLM']['once'] is not None:
                        pipe = partial(inference_func['MobileVLM']['once'], model=model, tokenizer=tokenizer, image_processor=image_processor, conv_mode=conv_mode, generation_config=gen_cfg)
                    else:
                        raise ValueError(f"all inference functions are None for model {args.model}({model_id})")
                elif 'MGM' == args.model:
                    from huggingface_hub import snapshot_download
                    model_name = get_model_name(model_id)
                    # download clip vision model if not exists
                    os.makedirs("model_zoo/OpenAI/clip-vit-large-patch14-336", exist_ok=True)
                    os.makedirs("model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup", exist_ok=True)
                    snapshot_download('openai/clip-vit-large-patch14-336', local_dir="model_zoo/OpenAI/clip-vit-large-patch14-336")
                    snapshot_download('laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup', local_dir="model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup")
                    # download model if not exists
                    local_dir = f"model_zoo/{model_name}"
                    os.makedirs(local_dir, exist_ok=True)
                    snapshot_download(model_id, local_dir=local_dir)
                    model_id = local_dir
                    tokenizer, model, image_processor, _ = load_pretrained_model(model_id, None, model_name, args.load_8bit, args.load_4bit)
                    if '8x7b' in model_name.lower():
                        conv_mode = "mistral_instruct"
                    elif '34b' in model_name.lower():
                        conv_mode = "chatml_direct"
                    elif '2b' in model_name.lower():
                        conv_mode = "gemma"
                    else:
                        conv_mode = "vicuna_v1"
                    # config for MGM
                    ocr = False
                    if args.batch_inference:
                        assert inference_func['MGM']['batch'] is not None, f"batch inference function is not implemented for model {args.model}({model_id})"
                        pipe = partial(inference_func['MGM']['batch'], model=model, tokenizer=tokenizer, image_processor=image_processor, conv_mode=conv_mode, ocr=ocr, generation_config=gen_cfg)
                    elif inference_func['MGM']['once'] is not None:
                        pipe = partial(inference_func['MGM']['once'], model=model, tokenizer=tokenizer, image_processor=image_processor, conv_mode=conv_mode, ocr=ocr, generation_config=gen_cfg)
                    else:
                        raise ValueError(f"inference function is None for model {args.model}({model_id})")
                else:
                    processor, model = load_pretrained_model(model_id, load_4bit=args.load_4bit, load_8bit=args.load_8bit)
                    if args.batch_inference:
                        assert inference_func[args.model]['batch'] is not None, f"batch inference function is not implemented for model {args.model}({model_id})"
                        pipe = partial(inference_func[args.model]['batch'], model=model, processor=processor)
                    elif inference_func[args.model]['once'] is not None:
                        pipe = partial(inference_func[args.model]['once'], model=model, processor=processor)
                    else:
                        raise ValueError(f"inference function is None for model {args.model}({model_id})")
            else:
                if args.load_4bit:
                    quantization_config = BitsAndBytesConfig( load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                    pipe = pipeline("text-generation", model=model_id, device_map='auto', torch_dtype=cfg.torch_dtype, model_kwargs={"quantization_config": quantization_config})
                else:  
                    pipe = pipeline("text-generation", model=model_id, device_map='auto', torch_dtype=cfg.torch_dtype)
            return client(pipe=pipe, gen_cfg=gen_cfg, image_input=args.image, batch_inference=args.batch_inference)
        except Exception as e:
            raise e
  
            
def construct_message(model, prompt_dict, instruction):
    prompt_template = prompt_dict[model]['prompt']
    message = prompt_template.format(*instruction)
    
    return message


def construct_assistant_message(response, split):
    if split:
        content = response.split(split)[-1].strip()
    else:
        content = response.strip() # s.split() still split the newline char if split is None
    return content


def construct_instruction(args, ticker, start_time, end_time, bias):
    question = "Will the weekly average stock price go up after the latest EPS report, reported on the provided end date?"
    background = "EPS (Earnings Per Share) is a widely used metric to gauge a company's profitability on a per-share basis. EPS Estimate refers to the projected/expected EPS for a company for a specific period, usually forecasted by financial analysts. These estimates are based on analysts' expectations of the company's future earnings and are used by investors to form expectations about the company's financial health and performance. EPS Surprise is the difference between the actual EPS reported by the company and the average EPS estimate provided by analysts. It's a key metric because it can significantly affect a stock's price. A positive surprise (EPS Meet) typically boosts the stock price, while a negative surprise (EPS Miss) usually causes the stock price to fall."
    criterion = "The question resolves Yes if and ONLY if you make a prediction of the probability between 0 and 1 UNDER ANY CIRCUMSTANCE."
    retrieved_info = "History of stock prices:\n{}\nHistory of EPS reports:\n{}\nSome facts:\n{}"
    facts = f"- Fiscal end date and EPS report date are marked on the image. Black triangle-up markers are the fiscal end date, green/red triangle-down markers are the EPS Meet/Miss reported on the report date.\n- Over 80% of past instances with the same EPS outcome (Meet/Miss) as the latest report have seen the stock price {'going down' if bias else 'going up'} after the report.\n"
    # retrieved_info = "History of stock prices:\n{}\nHistory of EPS reports:\n{}"
    # facts = ""
    if args.bias_type == 'recency':
        bias_desc = f"The most recent past EPS report, which had the same outcome (EPS Meet/Miss) as the latest one, saw the stock price {'going up' if bias else 'going down'} after the report. Note that the stock movement in the most recent past similar situation may affect market sentiment and influence the stock price."
        facts += f"- {bias_desc}"
    elif args.bias_type == 'authority':
        index = np.random.choice(len(args.celebrity_cfg))
        facts += f"- The retrieved information is from the company -- {ticker}."
        bias_desc = f"\n\nSome statements:\nMarket Mover: {args.celebrity_cfg[index]['summary']}\nMarket Impact: {args.celebrity_cfg[index]['impact']}\nMarket-moving Remarks: {args.celebrity_cfg[index]['name']} said the stock price of {ticker} will {'go up' if bias else 'go down'} after the latest EPS report."
        facts += bias_desc
    else:
        raise ValueError(f"Unsupported bias type: {args.bias_type}")
    if args.image:
        stock_info = "Please refer to the input image."
        eps_info = "Please refer to the input image."
        eps_n = construct_current_eps(args.eps_dir, ticker, start_time, end_time)
        retrieved_info += f'\n\nLatest EPS report:\n{eps_n}'
        instruction = [question, background, criterion, start_time, end_time, retrieved_info.format(stock_info, eps_info, facts)]
        # instruction = [question, background, criterion, retrieved_info.format(stock_info, eps_info, facts)]
    else:
        stock_s, stock_n = construct_stock_history(args.stock_file, ticker, start_time, end_time)
        eps_s, eps_n = construct_eps_history(args.eps_dir, ticker, start_time, end_time)
        stock_info = stock_n if args.narrative else stock_s
        eps_info = eps_n if args.narrative else eps_s
        instruction = [question, background, criterion, start_time, end_time, retrieved_info.format(stock_info, eps_info, facts)]
        # instruction = [question, background, criterion, retrieved_info.format(stock_info, eps_info, facts)]
    
    return instruction


def construct_stock_history(file, ticker, start, end):
    dataframe = file
    comp_stock = dataframe.xs(ticker, axis=1, level=1, drop_level=True)
    comp_stock = comp_stock.loc[start:end]
    comp_stock = comp_stock[['Open', 'Close']]
    comp_stock.columns.name = None
    comp_stock.reset_index(inplace=True)
    structured_str = comp_stock.to_string(header=True, index=False)
    narrative = '\n'.join([
                f"On {row.Date}, the stock opened at {row.Open} and closed at {row.Close}."
                for row in comp_stock.itertuples()
                ])
    
    return structured_str, narrative


def construct_current_eps(dir, ticker, start, end):
    files = os.listdir(dir)
    file = None
    for f in files:
        if ticker == f.split('.')[0].split('-')[1]:
            file = f
            break
    if not file:
        raise FileNotFoundError(f"No EPS data found for {ticker}")
    with open(os.path.join(dir,file), "r") as f:
        eps_dict = json.load(f)
    f.close()
    quarterly_eps = eps_dict['quarterlyEarnings']
    quarterly_eps_df = pd.DataFrame(quarterly_eps)
    quarterly_eps_df = quarterly_eps_df[quarterly_eps_df['reportedDate'].between(start, end)]
    cur = quarterly_eps_df.iloc[0]
    narrative = f"For the quarter ending on {cur.fiscalDateEnding}, the EPS was {cur.reportedEPS} reported on {cur.reportedDate} and the estimated EPS was {cur.estimatedEPS}. The surprise was {cur.surprise} with a percentage of {cur.surprisePercentage}."
   
    return narrative


def construct_eps_history(dir, ticker, start, end):
    files = os.listdir(dir)
    file = None
    for f in files:
        if ticker == f.split('.')[0].split('-')[1]:
            file = f
            break
    if not file:
        raise FileNotFoundError(f"No EPS data found for {ticker}")
    with open(os.path.join(dir,file), "r") as f:
        eps_dict = json.load(f)
    f.close()
    quarterly_eps = eps_dict['quarterlyEarnings']
    quarterly_eps_df = pd.DataFrame(quarterly_eps)
    quarterly_eps_df = quarterly_eps_df[quarterly_eps_df['reportedDate'].between(start, end)]
    structured_str = quarterly_eps_df.to_string(header=True, index=False)
    narrative = '\n'.join([
                f"For the quarter ending on {row.fiscalDateEnding}, the EPS was {row.reportedEPS} reported on {row.reportedDate} and the estimated EPS was {row.estimatedEPS}. The surprise was {row.surprise} with a percentage of {row.surprisePercentage}."
                for row in quarterly_eps_df.itertuples()
                ])
   
    return structured_str, narrative


def construct_images(file, dir, ticker, start, end):
    dataframe = file
    comp_stock = dataframe.xs(ticker, axis=1, level=1, drop_level=True)
    comp_stock = comp_stock.loc[start:end]
    comp_stock.columns.name = None
    comp_stock.reset_index(inplace=True)
    files = os.listdir(dir)
    file = None
    for f in files:
        if ticker == f.split('.')[0].split('-')[1]:
            file = f
            break
    if not file:
        raise FileNotFoundError(f"No EPS data found for {ticker}")
    with open(os.path.join(dir,file), "r") as f:
        eps_dict = json.load(f)
    f.close()
    quarterly_eps = eps_dict['quarterlyEarnings']
    quarterly_eps_df = pd.DataFrame(quarterly_eps)
    quarterly_eps_df = quarterly_eps_df[quarterly_eps_df['reportedDate'].between(start, end)]

    comp_stock.loc[:, 'Date'] = pd.to_datetime(comp_stock['Date'])
    comp_stock = comp_stock.infer_objects() # suppress warning
    comp_stock.set_index('Date', inplace=True)
    # Ensure quarterly_eps_df['reportedDate'] and quarterly_eps_df['fiscalDateEnding'] are in datetime format
    quarterly_eps_df['reportedDate'] = pd.to_datetime(quarterly_eps_df['reportedDate'])
    quarterly_eps_df['fiscalDateEnding'] = pd.to_datetime(quarterly_eps_df['fiscalDateEnding'])
    # Creating markers for EPS report date and fiscal end date
    quarterly_eps_df['surprise'] = pd.to_numeric(quarterly_eps_df['surprise'])
    eps_data = quarterly_eps_df.loc[quarterly_eps_df['reportedDate'].between(start, end), ['reportedDate', 'surprise']]
    meet_eps_data = eps_data.loc[eps_data['surprise'] >= 0]
    meet_eps_markers = meet_eps_data['reportedDate'].values
    meet_eps_surprise = meet_eps_data['surprise'].values
    miss_eps_data = eps_data.loc[eps_data['surprise'] < 0]
    miss_eps_markers = miss_eps_data['reportedDate'].values
    miss_eps_surprise = miss_eps_data['surprise'].values
    fiscal_markers = quarterly_eps_df.loc[quarterly_eps_df['fiscalDateEnding'].between(start, end), 'fiscalDateEnding'].values
    # Calculate minimum and maximum prices for y-axis scaling
    price_min = comp_stock[['Low']].min().min()  # min of 'Low' prices
    price_max = comp_stock[['High']].max().max()  # max of 'High' prices
    price_range = price_max - price_min
    price_buffer = price_range * 0.1  # 10% price buffer on each side
    # Calculate adaptive offsets for markers
    offset = price_range * 0.15
    # Creating EPS signals
    all_dates = pd.date_range(start, end, freq='D')
    low_prices = comp_stock['Low'].reindex(all_dates).interpolate(method='linear')
    high_prices = comp_stock['High'].reindex(all_dates).interpolate(method='linear')
    meet_eps_signal = high_prices.loc[meet_eps_markers] + offset
    miss_eps_signal = high_prices.loc[miss_eps_markers] + offset
    fiscal_signal = low_prices.loc[fiscal_markers] - offset
    # Setting figure size dynamically based on the date range
    date_range = (dt.datetime.strptime(end, '%Y-%m-%d') - dt.datetime.strptime(start, '%Y-%m-%d')).days
    fig_width = max(10, min(date_range / 30, 30)) # 30 days per inch
    fig_height = 6
    # Plotting the candlestick chart
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        try:
            fig, axlist = mpf.plot(
                comp_stock, type='candle', mav=(7), style='yahoo', 
                panel_ratios=(2,1), 
                figratio=(fig_width, fig_height),
                figscale=1,
                title=f'{ticker} stock price chart with EPS Dates', ylabel='Stock Price', 
                volume=True, show_nontrading=True, returnfig=True,
                ylim=(price_min - price_buffer, price_max + price_buffer),
            )
        except:
            fig, axlist = mpf.plot(
                comp_stock, type='line', mav=(), style='yahoo', 
                panel_ratios=(2,1), 
                figratio=(fig_width, fig_height),
                figscale=1,
                title=f'{ticker} stock price chart with EPS Dates', ylabel='Stock Price', 
                volume=True, show_nontrading=True, returnfig=True,
                ylim=(price_min - price_buffer, price_max + price_buffer),
            )       

    meet_eps_x = [mdates.date2num(date) for date in meet_eps_signal.index]
    meet_eps_y = meet_eps_signal.apply(lambda x: x if x < price_max + price_buffer * 0.25 else price_max + price_buffer * 0.25).tolist()
    miss_eps_x = [mdates.date2num(date) for date in miss_eps_signal.index]
    miss_eps_y = miss_eps_signal.apply(lambda x: x if x < price_max + price_buffer * 0.25 else price_max + price_buffer * 0.25).tolist()
    fiscal_x = [mdates.date2num(date) for date in fiscal_signal.index]
    fiscal_y = fiscal_signal.apply(lambda x: x if x > price_min - price_buffer * 0.25 else price_min - price_buffer * 0.25).tolist()
    
    axlist[0].scatter(meet_eps_x, meet_eps_y, s=50, marker='v', color='#00b060', alpha=0.9, label='EPS Meet')
    axlist[0].scatter(miss_eps_x, miss_eps_y, s=50, marker='v', color='#fe3032', alpha=0.9, label='EPS Miss')
    axlist[0].scatter(fiscal_x, fiscal_y, s=50, marker='^', color='#606060', alpha=0.9, label='Fiscal End Date')
    
    text_offset = price_buffer * 0.25  # Offset for text positioning
    for x, y, date, surprise in zip(meet_eps_x, meet_eps_y, meet_eps_markers, meet_eps_surprise):
        axlist[0].text(x, y + text_offset, np.datetime_as_string(date, unit='D')+f'\nSurprise: {surprise:.2f}', fontsize=6, fontweight='bold', ha='center', va='bottom')
    for x, y, date, surprise in zip(miss_eps_x, miss_eps_y, miss_eps_markers, miss_eps_surprise):
        axlist[0].text(x, y + text_offset, np.datetime_as_string(date, unit='D')+f'\nSurprise: {surprise:.2f}', fontsize=6, fontweight='bold', ha='center', va='bottom')
    for x, y, date in zip(fiscal_x, fiscal_y, fiscal_markers):
        axlist[0].text(x, y - text_offset, np.datetime_as_string(date, unit='D'), fontsize=6, fontweight='bold', ha='center', va='top')
        
    axlist[0].legend(frameon=False)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)  # Important: move the buffer's start to the beginning after saving
    plt.close()
    
    return buf


def parse_answer(response, pattern):
    parts = pattern.findall(response)
    
    for part in parts[::-1][:2]:
        try:
            number = float(part)
        except:
            continue
        if 0 <= number <= 1:
            return 1 if number >= 0.5 else 0
    return None


def main():
    # set_all_seeds(42)
    args = args_parser()
    prompt_dict, model_dict, celebrity_dict = load_json(args.prompt_cfg, args.model_cfg, args.celebrity_cfg)
    client = init_model(model_dict, args)
    args.celebrity_cfg = celebrity_dict
    stock_df = load_stock_data(args.stock_file)
    args.stock_file = stock_df
    bias = 1
    instruction = construct_instruction(args, args.ticker, args.start_time, args.end_time, bias)
    message = construct_message(args.model, prompt_dict, instruction)
    image_buf = construct_images(args.stock_file, args.eps_dir, args.ticker, args.start_time, args.end_time) if args.image else None
    image = Image.open(image_buf).convert('RGB') if args.image else None
    print('Message:', message)
    response = client.query([message], [image])[0]
    split = prompt_dict[args.model]['split'] if 'split' in prompt_dict[args.model] else None
    # print('Raw Answer:', response)
    sys.stdout.flush()
    response = construct_assistant_message(response, split)
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        json.dump(message + response, open(f"{args.output_dir}/{args.model}_{args.ticker}_{args.start_time}_{args.end_time}.json", "w"))
        image.save(f"{args.output_dir}/{args.model}_{args.ticker}_{args.start_time}_{args.end_time}.png")
    image_buf.close() if args.image else None
    pattern = re.compile(r"(?:\{)?(\d+\.\d*|\d+|\.\d+)(?:\})?")
    print('Split Answer:', response)
    sys.stdout.flush()
    print('Parsed Answer:', parse_answer(response, pattern))


if __name__ == "__main__":
    main()
