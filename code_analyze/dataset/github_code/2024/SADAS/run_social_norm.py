import openai
# import ai21
import re
import time
import json
import sys
import copy
import random

import numpy as np

from tqdm import tqdm
from pprint import pprint

import keys
from agent import (load_initial_instructions, involve_moderator, parse_final_price, 
    BuyerAgent, SellerAgent, ModeratorAgent, SellerCriticAgent, BuyerCriticAgent, RemediatorAgent, load_initial_instructions_withprefix)
from utils import *

from retry.api import retry_call

STOP_AFTER_ATTEMPT=4

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

CONST_CRITIC_PATH = "lib_prompt/balloon/constant_feedback.txt"
HUMAN_CRITIC_PATH = "lib_prompt/balloon/human_feedback_seller.txt"

dict_ = {'buyer': set(), 'seller': set()}

import argparse

# todo: we need absolute violate/no-violate for selecting the ICL examples
def define_arguments():
    parser = argparse.ArgumentParser()

    # price related to the house
    parser.add_argument('--commodity', type=str, default="industrial", help="[balloon, salary, house, industrial, business]")

    # # business
    # parser.add_argument('--cost_price', type=int, default=0,
    #                     help='cost of the store')
    # parser.add_argument('--seller_init_price', type=int, default=0,
    #                     help='initial offered price')
    # parser.add_argument('--buyer_init_price', type=int, default=0,
    #                     help='initial required price')

    # industrial
    parser.add_argument('--cost_price', type=int, default=10,
                        help='cost of the unit price of an industrial commodity')
    parser.add_argument('--seller_init_price', type=int, default=50,
                        help='initial offered price')
    parser.add_argument('--buyer_init_price', type=int, default=30,
                        help='initial required price')

    # # house
    # parser.add_argument('--cost_price', type=int, default=600000,
    #                     help='cost of the house')
    # parser.add_argument('--seller_init_price', type=int, default=700000,
    #                     help='initial offered price')
    # parser.add_argument('--buyer_init_price', type=int, default=500000,
    #                     help='initial required price')

    # # price related to the salary
    # parser.add_argument('--commodity', type=str, default="salary", help="[balloon, salary, house]")
    # parser.add_argument('--cost_price', type=int, default=5000,
    #                     help='Budget of the salary')
    # parser.add_argument('--seller_init_price', type=int, default=4000,
    #                     help='initial offered salary')
    # parser.add_argument('--buyer_init_price', type=int, default=3000,
    #                     help='initial required price')

    # # price related to the balloon
    # parser.add_argument('--commodity', type=str, default="balloon", help="[balloon, salary, house]")
    # parser.add_argument('--cost_price', type=int, default=8,
    #                     help='Cost of the baloon')
    # parser.add_argument('--seller_init_price', type=int, default=20,
    #                     help='initial seller price')
    # parser.add_argument('--buyer_init_price', type=int, default=10,
    #                     help='initial buyer price')

    # seller arguments
    parser.add_argument('--seller_engine', type=str, default="gpt-3.5-turbo")
    # parser.add_argument('--seller_instruction', type=str, default="seller_cn")
    # parser.add_argument('--seller_instruction', type=str, default="seller_extraversion")
    # parser.add_argument('--seller_instruction', type=str, default="seller_agreeableness")
    # parser.add_argument('--seller_instruction', type=str, default="seller_extraversion_agreeableness")
    # parser.add_argument('--seller_instruction', type=str, default="seller_social_norm_cn")
    # parser.add_argument('--seller_instruction', type=str, default="seller_social_norm_explicit_labeling_cn")
    # parser.add_argument('--seller_instruction', type=str, default="seller_social_norm_explicit_labeling_cn_withgoals")
    parser.add_argument('--seller_instruction', type=str, default="seller_social_norm_explicit_labeling_cn_withgoals_absolute_violate")

    parser.add_argument('--reset_seller_initial_history_every_exp', type=bool, default=True,
                        help='generate social norm instruct rules every experiment, used for prefix generation')
    parser.add_argument('--seller_prefix_instruction', type=str, default="seller_social_prefix_cn_simplified_withgoals", help="[seller_social_prefix_cn_simplified, seller_social_prefix_cn, seller_social_prefix_cn_simplified_withgoals]")

    parser.add_argument('--seller_critic_engine', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--seller_critic_instruction', type=str, default="seller_critic")

    # buyer arguments
    parser.add_argument('--buyer_engine', type=str, default="gpt-3.5-turbo")
    # parser.add_argument('--buyer_instruction', type=str, default="buyer_with_face",
    #                     help="[buyer, buyer_no_initial_price, buyer_with_face]")
    parser.add_argument('--buyer_instruction', type=str, default="buyer_cn_withgoals",
                        help="[buyer, buyer_no_initial_price, buyer_with_face, buyer_cn, buyer_cn_withgoals]")

    parser.add_argument('--buyer_critic_engine', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--buyer_critic_instruction', type=str, default="buyer_critic",
                        help="[buyer_critic, buyer_critic_no_initial_price]")

    # moderator arguments
    # parser.add_argument('--moderator_instruction', type=str, default="moderator_0509_with_face_norm",
    #                     help="[moderator_0509, moderator_buyer, moderator_seller, moderator_buyer_reason_first]")
    parser.add_argument('--moderator_instruction', type=str, default="moderator_cn",
                        help="[moderator_cn, moderator, moderator_0509, moderator_buyer, moderator_seller, moderator_buyer_reason_first]")
    parser.add_argument('--moderator_engine', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--moderator_trace_n_history', type=int, default=5,
                        help="how long the moderator trace history")

    # api keys
    # parser.add_argument('--api_key', type=str, default=None, help='openai api key')
    parser.add_argument('--api_key', type=str, default=random.choice(keys.keys), help='openai api key')
    parser.add_argument('--anthropic_api_key', type=str, default=None, help='anthropic api key')
    parser.add_argument('--ai21_api_key', type=str, default=None, help='ai21 api key')
    parser.add_argument('--cohere_api_key', type=str, default=None, help='cohere api key')

    # game arguments
    parser.add_argument('--game_type', type=str, default='run_simple',
                        help='[criticize_seller, criticize_buyer, seller_compare_feedback, run_simple]')
    parser.add_argument('--n_exp', type=int, default=50,
                        help='number of experiments')
    parser.add_argument('--n_round', type=int, default=20,
                        help='number of rounds, 20 for business bargain')
    parser.add_argument('--n_rollout', type=int, default=5,
                        help='number of rollout')

    parser.add_argument('--verbose', type=int, default=1, help="0: not logger.write, 1: logger.write")
    parser.add_argument('--output_path', type=str, default="./outputs/",
                        help='path to save the output')
    parser.add_argument('--ver', type=str, default="criticize_gpt3.5_seller",
                        help='version to record the game')
    parser.add_argument('--game_version', type=str, default="test", 
                        help='version plus arguments')

    parser.add_argument('--remediator', type=bool, default=True,
                        help='call the remediator to remedy the norm violation')
    parser.add_argument('--remediator_instruction', type=str, default="remediator_cn",
                        help="[remediator_cn]")
    parser.add_argument('--remediator_engine', type=str, default="gpt-3.5-turbo", help='[gpt-3.5-turbo, atom-7b-chat]')

    # parse and set arguments
    args = parser.parse_args()

    openai.api_key = args.api_key
    # ai21.api_key = args.ai21_api_key
    return args

def get_engine_and_api_key(agent_type, engine_name, args):
    """Get engine for players and critic
    agent_type: [seller, buyer, seller_critic, buyer_critic, moderator]
    engine_name: [gpt-3.5-turbo, gpt-4, claude-v1.0, claude-v1.3, claude-instant-v1.0]
    """
    engine_map = {  "seller": SellerAgent, 
                    "buyer":  BuyerAgent, 
                    "seller_critic": SellerCriticAgent, 
                    "buyer_critic":  BuyerCriticAgent,
                    "moderator": ModeratorAgent,
                    "remediator": RemediatorAgent
                  }

    if("gpt" in engine_name): 
        api_key = args.api_key
    elif("claude" in engine_name): 
        api_key = args.anthropic_api_key
    elif("j2" in engine_name):
        api_key = args.ai21_api_key
    elif("cohere" in engine_name):
        api_key = args.cohere_api_key
    elif("atom" in engine_name):
        api_key = ""
    else:
        raise ValueError("engine name %s not found" % engine_name)

    engine_class = engine_map[agent_type]

    return engine_class, api_key

def call_engine_for_generating_prefix(prefix_dialog_history, args):
    try:
        completion = retry_call(openai.ChatCompletion.create, fkwargs={"model":"gpt-3.5-turbo",
            "messages":prefix_dialog_history,
            "api_key":args.api_key,
            "n":1,"temperature":1.0, "request_timeout":30}, tries=3, delay=1, jitter=1)
        for choice in completion['choices']:
            # print("====================================")
            chatgpt_message = choice['message']['content']
            print('chatgpt_message is: {}'.format(chatgpt_message))
            return completion['choices'][0]['message']['content']
    except:
        print('-----------------------------openai API is failed!!!!!------------------------------------')
        completion = {'choices':[{'message':{'content':'Error'}}]}
        return 'null'

def extract_social_norm_rules(res_):
    if "1." not in res_:
        return "1. 例子：使用侮辱性语言 2. 例子：打断别人的发言。 3. 例子：嘲笑或讽刺他人。4. 例子：忽视他人的观点或感受。 5. 例子：过度争论或争执。 "
    else:
        return res_

def run(buyer, seller, moderator, 
        n_round=10, who_is_first="seller", no_deal_thres=10, args=None, remediator=None):
    """Run single game.
    """
    logger.write('---- start bargaining ----')

    # todo only first sentence has violation, using COT?

    if(who_is_first == "buyer"): # buyer starts the first LLM generation
        logger.write('  buyer: %s' % buyer.last_response)
        dict_['buyer'].add(buyer.last_response)
        seller_run = seller.last_response
        buyer_run = buyer.call(seller_run)
        if remediator is not None and ('【violation】' in buyer.dialog_history[-1]['content'] or
                                       '【违规】' in buyer.dialog_history[-1]['content'] or
                                       '【违反社会规范】' in buyer.dialog_history[-1]['content'] or
                                       '【违反规范】' in buyer.dialog_history[-1]['content'] or
                                       'violation' in buyer.dialog_history[-1]['content'] or
                                       '违规' in buyer.dialog_history[-1]['content']):
            logger.write('【Before remediation】: %s' % buyer.dialog_history[-1]['content'])
            sentence = remediator.produce_remediation(buyer.dialog_history)
            buyer.remediate_conversation(sentence)
            logger.write('【After remediation】: %s' % buyer.dialog_history[-1]['content'])
            buyer_run = sentence

    logger.write('  seller: %s' % seller.last_response)
    logger.write('  buyer: %s' % buyer.last_response)
    dict_['seller'].add(seller.last_response)
    dict_['buyer'].add(buyer.last_response)

    buyer_run = buyer.last_response
    start_involve_moderator = False
    deal_at = "none"
    no_deal_cnt = 0
    for _ in range(n_round):
        seller_run = seller.call(buyer_run)
        if remediator is not None and ('【violation】' in seller.dialog_history[-1]['content'] or
                                       '【违规】' in seller.dialog_history[-1]['content'] or
                                       '【违反社会规范】' in seller.dialog_history[-1]['content'] or
                                       '【违反规范】' in seller.dialog_history[-1]['content'] or
                                       'violation' in seller.dialog_history[-1]['content'] or
                                       '违规' in seller.dialog_history[-1]['content']):
            logger.write('【Before remediation】: %s' % seller.dialog_history[-1]['content'])
            sentence = remediator.produce_remediation(seller.dialog_history)
            seller.remediate_conversation(sentence)
            logger.write('【After remediation】: %s' % seller.dialog_history[-1]['content'])
            seller_run = sentence
        logger.write('  seller: %s' % seller.last_response)
        dict_['seller'].add(seller.last_response)
        
        if(start_involve_moderator is False and involve_moderator(buyer_run, seller_run)):
            start_involve_moderator = True
            logger.write('---- start moderating ----')
        
        if(start_involve_moderator):
            moderate = moderator.moderate(seller.dialog_history, who_was_last="seller")
            logger.write('MODERATE have the seller and the buyer achieved a deal? Yes or No: %s' % moderate)
            if("yes" in moderate.lower()): 
                deal_at = "seller"
                break
            else: 
                no_deal_cnt += 1
                if(no_deal_cnt == no_deal_thres): break
            
        buyer_run = buyer.call(seller_run)
        if remediator is not None and ('【violation】' in buyer.dialog_history[-1]['content'] or
                                       '【违规】' in buyer.dialog_history[-1]['content'] or
                                       '【违反社会规范】' in buyer.dialog_history[-1]['content'] or
                                       '【违反规范】' in buyer.dialog_history[-1]['content'] or
                                       'violation' in buyer.dialog_history[-1]['content'] or
                                       '违规' in buyer.dialog_history[-1]['content']):
            logger.write('【Before remediation】: %s' % buyer.dialog_history[-1]['content'])
            sentence = remediator.produce_remediation(buyer.dialog_history)
            buyer.remediate_conversation(sentence)
            logger.write('【After remediation】: %s' % buyer.dialog_history[-1]['content'])
            buyer_run = sentence
        logger.write('  buyer: %s' % buyer.last_response)
        dict_['buyer'].add(buyer.last_response)
        
        if(start_involve_moderator is False and involve_moderator(buyer_run, seller_run)):
            start_involve_moderator = True
            logger.write('---- start moderating ----')
            
        if(start_involve_moderator):
            moderate = moderator.moderate(buyer.dialog_history, who_was_last="buyer")
            logger.write('MODERATE have the seller and the buyer achieved a deal? Yes or No: %s' % moderate)
            if("yes" in moderate.lower()): 
                deal_at = "buyer"
                break
            else: 
                no_deal_cnt += 1
                if(no_deal_cnt == no_deal_thres): break
                
    if(deal_at != "none"):
        if(deal_at == "seller"):
            final_price = parse_final_price(seller.dialog_history)
        else: 
            final_price = parse_final_price(buyer.dialog_history)
        return final_price
    else: return -1
    
def run_compare_critic_single(buyer, seller, moderator, critic,
                       const_feedback, human_feedback_pool, 
                       game_type, n_round=10, who_is_first="seller"):
    """Run with multiple types of critic then compare the effect of different critics
    """
    logger.write('==== RUN 1 ====')
    buyer.reset()
    seller.reset()
    moderator.reset()
    run_n_prices, run_n_prices_const, run_n_prices_human = [], [], [] 

    run_1_price = run(buyer, seller, moderator, n_round=n_round, who_is_first=who_is_first)
    logger.write('PRICE: %s' % run_1_price)
    run_n_prices.append(run_1_price)
    run_n_prices_const.append(run_1_price)
    run_n_prices_human.append(run_1_price)
    
    # Round 2 after critic
    logger.write('==== RUN 2 ====')

    if(game_type == "seller_compare_feedback"):
        seller_hear_const = copy.deepcopy(seller)
        seller_hear_human = copy.deepcopy(seller)

        # ai feedback 
        buyer.reset()
        moderator.reset()

        ai_feedback = critic.criticize(seller.dialog_history)
        
        logger.write("AI FEEDBACK:\n%s\n" % ai_feedback)
        acknowledgement = seller.receive_feedback(ai_feedback, run_1_price)
        logger.write("ACK:\n%s\n\n" % acknowledgement)
        run_2_price = run(buyer, seller, moderator, n_round=n_round, who_is_first=who_is_first)
        logger.write('PRICE: %s' % run_2_price)
        run_n_prices.append(run_2_price)

        # const feedback
        buyer.reset()
        moderator.reset()
        logger.write("\n\nCONST FEEDBACK:\n%s\n" % const_feedback)
        acknowledgement = seller_hear_const.receive_feedback(const_feedback, run_1_price)
        logger.write("ACK:\n%s\n\n" % acknowledgement)
        run_2_price = run(buyer, seller_hear_const, moderator, n_round=n_round, who_is_first=who_is_first)
        logger.write('PRICE: %s' % run_2_price)
        run_n_prices_const.append(run_2_price)

        # human feedback
        buyer.reset()
        moderator.reset()
        human_feedback = random.choice(human_feedback_pool)
        logger.write("\n\nHUMAN FEEDBACK:\n%s\n" % human_feedback)
        acknowledgement = seller_hear_human.receive_feedback(human_feedback, run_1_price)
        logger.write("ACK:\n%s\n\n" % acknowledgement)
        run_2_price = run(buyer, seller_hear_human, moderator, n_round=n_round, who_is_first=who_is_first)
        logger.write('PRICE: %s' % run_2_price)
        run_n_prices_human.append(run_2_price)

    elif(game_type == "buyer_compare_feedback"):
        raise NotImplementedError("buyer_compare_feedback not implemented yet")
    else: raise ValueError("game_type must be either 'critize_seller' or 'critize_buyer'")
    
    return run_n_prices, run_n_prices_const, run_n_prices_human

def run_compare_critic(args, buyer, seller, moderator, critic, 
                            const_feedback, human_feedback_pool, 
                            game_type,
                            n_exp=100, n_round=10, who_is_first="seller"):
    """run multiple experiments with multiple types of critic
    """
    prices_ai_critic, prices_const_critic, prices_human_critic = [], [], []

    start_time = time.time()
    for i in range(n_exp):
        logger.write("==== ver %s CASE %d / %d, %.2f min ====" % (args.ver, i, n_exp, compute_time(start_time)))
        buyer.reset()
        seller.reset()
        moderator.reset()
        ai_price, const_price, human_price = run_compare_critic_single(buyer, seller, moderator, critic, 
                                            const_feedback, human_feedback_pool,
                                            game_type=game_type, 
                                            n_round=n_round, who_is_first=who_is_first)
        
        if(check_k_price_range(ai_price, p_min=args.buyer_init_price, p_max=args.seller_init_price) and 
           check_k_price_range(const_price, p_min=args.buyer_init_price, p_max=args.seller_init_price) and 
           check_k_price_range(human_price, p_min=args.buyer_init_price, p_max=args.seller_init_price)
           ):
            prices_ai_critic.append(ai_price)
            prices_const_critic.append(const_price)
            prices_human_critic.append(human_price)
            assert(ai_price[0] == const_price[0] == human_price[0])
        logger.write("\n\n\n\n")

    round_0_price = [price[0] for price in prices_ai_critic]
    round_1_price_ai_critic = [price[1] for price in prices_ai_critic]
    round_1_price_const_critic = [price[1] for price in prices_const_critic]
    round_1_price_human_critic = [price[1] for price in prices_human_critic]

    logger.write("Round 0 price:              %.2f std: %.2f" % 
                 (np.mean(round_0_price), np.std(round_0_price))
                 )
    logger.write("Round 1 price ai critic:    %.2f std: %.2f" % 
                 (np.mean(round_1_price_ai_critic), np.std(round_1_price_ai_critic))
                 )
    logger.write("Round 1 price const critic: %.2f std: %.2f" % 
                 (np.mean(round_1_price_const_critic), np.std(round_1_price_const_critic))
                 )
    logger.write("Round 1 price human critic: %.2f std: %.2f" % 
                 (np.mean(round_1_price_human_critic), np.std(round_1_price_human_critic))
                 )
    logger.write("%d runs, %d effective" % (n_exp, len(prices_ai_critic)))
    return 


def run_simple(args, buyer, seller, moderator,
                n_exp=100, n_round=10, who_is_first="seller"):
    """run multiple experiments without critic, simply checking if the model can play the game
    """
    # print('who_is_first is {}'.format(who_is_first))
    start_time = time.time()
    sum_price = 0.0
    effective_price_num = 0
    deal_prices = []
    remediator = None

    if args.remediator is True:
        remediator_initial_dialog_history = load_initial_instructions(
            "lib_prompt/{}/{}.txt".format(args.commodity, args.remediator_instruction))
        remediator_engine_class, remediator_api_key = get_engine_and_api_key(engine_name=args.remediator_engine,
                                                                           agent_type="remediator", args=args
                                                                           )
        remediator = remediator_engine_class(initial_dialog_history=remediator_initial_dialog_history,
                                           agent_type="remediator", engine=args.remediator_engine,
                                           api_key=remediator_api_key
                                           )

    if args.reset_seller_initial_history_every_exp is not True:
        seller_initial_dialog_history = load_initial_instructions(
            'lib_prompt/{}/{}.txt'.format(args.commodity, args.seller_instruction))
        seller.reset_initial_dialogue_history(seller_initial_dialog_history)

    for i in range(n_exp):
        logger.write("==== ver %s CASE %d / %d, %.2f min ====" % (args.ver, i, n_exp, compute_time(start_time)))

        if args.reset_seller_initial_history_every_exp is True:
            # seller init
            prefix_dialog_history = load_initial_instructions(
                "lib_prompt/{}/{}.txt".format(args.commodity, args.seller_prefix_instruction))
            res_ = call_engine_for_generating_prefix(prefix_dialog_history, args)
            social_norm_rules = extract_social_norm_rules(res_)

            # todo do we have prefix in the dialogue history or not?
            seller_initial_dialog_history = load_initial_instructions_withprefix(
                'lib_prompt/{}/{}.txt'.format(args.commodity, args.seller_instruction), social_norm_rules)
            seller.reset_initial_dialogue_history(seller_initial_dialog_history)
        # set price
        buyer.reset()
        seller.reset()
        moderator.reset()

        logger.write('== The seller\'s initial prompt ==')
        logger.write('{}'.format(seller.dialog_history))
        logger.write('== The buyer\'s initial prompt ==')
        logger.write('{}'.format(buyer.dialog_history))
        price = run(buyer, seller, moderator, who_is_first=who_is_first, args=args, remediator=remediator)
        deal_prices.append(price)
        if price != -1:
            effective_price_num += 1
            sum_price += price
            logger.write("The final deal price is {}".format(price))
        else:
            logger.write("The deal is BROKEN!")
        logger.write("\n\n\n\n")
    average_price = 0.0 if effective_price_num == 0 else float(sum_price/effective_price_num)
    logger.write("\nSuccessful deal rate: {}%, average price: {}".format(float(effective_price_num/n_exp*100.0), average_price))
    logger.write("\nDeal prices are: {}".format(' '.join([str(x) for x in deal_prices]).strip()))
    return

def run_w_critic_rollout(args, buyer, seller, moderator, critic, game_type, 
                            n_rollout=3,
                            n_round=10, who_is_first="seller"):
    """Run multiple rounds of bargaining with one single critic
    """
    logger.write('==== RUN 1 ====')
    buyer.reset()
    seller.reset()
    moderator.reset()
    run_n_prices = []

    run_1_price = run(buyer, seller, moderator, n_round=n_round, who_is_first=who_is_first)
    logger.write('PRICE: %s' % run_1_price)
    run_n_prices.append(run_1_price)
    previous_price = run_1_price
    for i in range(n_rollout - 1):
        # Round i after critic
        if(game_type == "criticize_seller"):
            buyer.reset()
        elif(game_type == "criticize_buyer"):
            seller.reset()
        else: raise ValueError("game_type must be either 'critize_seller' or 'critize_buyer'")

        moderator.reset()
        if(game_type == "criticize_seller"):
            ai_feedback = critic.criticize(seller.dialog_history)
            logger.write("FEEDBACK:\n%s\n\n" % ai_feedback)
            acknowledgement = seller.receive_feedback(ai_feedback, previous_price)
            logger.write("ACK:\n%s\n\n" % acknowledgement)
        elif(game_type == "criticize_buyer"):
            ai_feedback = critic.criticize(buyer.dialog_history)
            logger.write("FEEDBACK:\n%s\n\n" % ai_feedback)
            acknowledgement = buyer.receive_feedback(ai_feedback, previous_price)
            logger.write("ACK:\n%s\n\n" % acknowledgement)
        else: raise ValueError("game_type must be either 'critize_seller' or 'critize_buyer'")
        
        logger.write('==== RUN %d ====' % (i + 2))
        run_i_price = run(buyer, seller, moderator, n_round=n_round, who_is_first=who_is_first)
        logger.write('PRICE: %s' % run_i_price)

        if(check_price_range(
            run_i_price, p_min=args.buyer_init_price, p_max=args.seller_init_price) == False
            ):
            logger.write("run %d did not get a deal, stop unroll" % (i + 2))
            break

        run_n_prices.append(run_i_price)
        previous_price = run_i_price
    return run_n_prices

def run_with_critic(args, buyer, seller, moderator, critic, game_type,
                            n_exp=100, n_rollout=3, n_round=10, who_is_first="seller"):
    """run multiple experiments with one single critic
    """
    round_k_prices = {k: [] for k in range(n_rollout)}

    # for i in tqdm(range(n_exp)):
    start_time = time.time()
    for i in range(n_exp):
        logger.write("==== ver %s CASE %d / %d | %.2f min ====" % (args.ver, i, n_exp, compute_time(start_time)))
        buyer.reset()
        seller.reset()
        moderator.reset()
        round_prices = run_w_critic_rollout(args, buyer, seller, moderator, critic, game_type, 
                                            n_rollout=n_rollout,
                                            n_round=n_round, who_is_first=who_is_first)
        # if(len(round_prices) == 5):
        
        # if(check_k_price_range(round_prices)):
            # for k in range(n_rollout):
            # for k in range(len(round_prices)):
            #     round_k_prices[k].append(float(round_prices[k]))
            #     logger.write("%.2f" % round_prices[k])

        logger.write("Price trace:")
        for k in range(len(round_prices)):
            round_k_prices[k].append(float(round_prices[k]))
            logger.write("%.2f" % round_prices[k])
        logger.write("\n\n\n\n")

    for k in range(n_rollout):
        logger.write("Round %d price: %.2f std: %.2f" % (k+1, np.mean(round_k_prices[k]), np.std(round_k_prices[k])))
    logger.write("%d runs, %d effective" % (n_exp, len(round_k_prices[0])))
    return 


def main(args):
    # # seller init
    # prefix_dialog_history = load_initial_instructions(
    #     "lib_prompt/{}/{}.txt".format(args.commodity, args.seller_prefix_instruction))
    # res_ = call_engine_for_generating_prefix(prefix_dialog_history, args)
    # social_norm_rules = extract_social_norm_rules(res_)
    #
    # seller_initial_dialog_history = load_initial_instructions_withprefix('lib_prompt/{}/{}.txt'.format(args.commodity, args.seller_instruction), social_norm_rules)

    logger.write('commodity: {}'.format(args.commodity))
    logger.write('seller_instruction: {}'.format(args.seller_instruction))
    logger.write('reset_seller_initial_history_every_exp: {}'.format(args.reset_seller_initial_history_every_exp))
    logger.write('seller_prefix_instruction: {}'.format(args.seller_prefix_instruction))
    logger.write('buyer_instruction: {}'.format(args.buyer_instruction))
    logger.write('moderator_instruction: {}'.format(args.moderator_instruction))
    logger.write('game_type: {}'.format(args.game_type))
    logger.write('n_exp: {}'.format(args.n_exp))
    logger.write('n_rollout: {}'.format(args.n_rollout))
    logger.write('ver: {}'.format(args.ver))
    logger.write('remediator: {}'.format(args.remediator))
    logger.write('remediator_instruction: {}'.format(args.remediator_instruction))
    logger.write('remediator_engine: {}'.format(args.remediator_engine))


    seller_engine_class, seller_api_key = get_engine_and_api_key(engine_name=args.seller_engine,
                                                            agent_type="seller", args=args
                                                            )
    seller = seller_engine_class(initial_dialog_history=None,
                                agent_type="seller", engine=args.seller_engine, api_key=seller_api_key,
                                cost_price=args.cost_price, 
                                buyer_init_price=args.buyer_init_price,
                                seller_init_price=args.seller_init_price
                                )

    # buyer init
    buyer_initial_dialog_history = load_initial_instructions('lib_prompt/{}/{}.txt'.format(args.commodity, args.buyer_instruction))
    buyer_engine_class, buyer_api_key = get_engine_and_api_key(engine_name=args.buyer_engine,
                                                            agent_type="buyer", args=args
                                                            )
    buyer = buyer_engine_class(initial_dialog_history=buyer_initial_dialog_history, 
                                agent_type="buyer", engine=args.buyer_engine, api_key=buyer_api_key,
                                buyer_instruction=args.buyer_instruction,
                                buyer_init_price=args.buyer_init_price,
                                seller_init_price=args.seller_init_price,
                                cost_price=args.cost_price
                                )


    # moderator init 
    moderator_initial_dialog_history = load_initial_instructions("lib_prompt/{}/{}.txt".format(args.commodity, args.moderator_instruction))
    moderator_engine_class, moderator_api_key = get_engine_and_api_key(engine_name=args.moderator_engine,
                                                                    agent_type="moderator", args=args
                                                                    )
    moderator = moderator_engine_class(initial_dialog_history=moderator_initial_dialog_history, 
                            agent_type="moderator", engine=args.moderator_engine, api_key=moderator_api_key,
                            trace_n_history=args.moderator_trace_n_history
                            )

    # critic init 
    if(args.game_type in ["criticize_seller", "seller_compare_feedback"]): 
         # seller critic init
        seller_critic_initial_dialog_history = load_initial_instructions('lib_prompt/%s.txt' % args.seller_critic_instruction)
        seller_critic_engine_class, seller_critic_api_key = get_engine_and_api_key(engine_name=args.seller_critic_engine,
                                                                                agent_type="seller_critic", args=args
                                                                                )
        seller_critic = seller_critic_engine_class(initial_dialog_history=seller_critic_initial_dialog_history, 
                                    agent_type="critic", engine=args.seller_critic_engine, api_key=seller_critic_api_key
                                    )
        critic = seller_critic
    elif(args.game_type == "criticize_buyer"): 
        # buyer critic init
        buyer_critic_initial_dialog_history = load_initial_instructions('lib_prompt/%s.txt' % args.buyer_critic_instruction)
        buyer_critic_engine_class, buyer_api_key = get_engine_and_api_key(engine_name=args.buyer_critic_engine,
                                                                        agent_type="buyer_critic", args=args
                                                                        )
        buyer_critic = buyer_critic_engine_class(initial_dialog_history=buyer_critic_initial_dialog_history, 
                                    agent_type="critic", engine=args.buyer_critic_engine, api_key=buyer_api_key
                                    )
        critic = buyer_critic
    elif(args.game_type == "run_simple"): pass
    else: raise ValueError("game_type must be in ['criticize_seller', 'criticize_buyer', 'seller_compare_feedback', 'run_simple']")

    # run
    if args.commodity == "salary":
        who_is_first = "buyer"
    else:
        who_is_first = "seller"
    if(args.buyer_instruction == "buyer_no_initial_price"):
        who_is_first = "buyer"

    if(args.game_type in ["criticize_seller", "criticize_buyer"]):
        run_with_critic(args, buyer, seller, moderator, critic, 
                                game_type=args.game_type, n_exp=args.n_exp, 
                                n_rollout=args.n_rollout, n_round=args.n_round,
                                who_is_first=who_is_first)
    elif(args.game_type == "seller_compare_feedback"):
        const_feedback = open(CONST_CRITIC_PATH).read().strip()
        human_feedback_pool = open(HUMAN_CRITIC_PATH).read().strip().split("\n")
        run_compare_critic(args, buyer, seller, moderator, critic, 
                            const_feedback, human_feedback_pool, 
                            game_type=args.game_type, n_exp=args.n_exp, 
                            n_round=args.n_round,
                            who_is_first=who_is_first)
    elif(args.game_type == "run_simple"):
        run_simple(args, buyer, seller, moderator, n_exp=args.n_exp, who_is_first=who_is_first)
    return 

def print_wordcloud(set_, image_path):
    directory_path = os.path.dirname(image_path)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Create a Counter object
    counter_ = Counter()

    list_ = list(set_)
    # print(list_)

    for x in list_:
        counter_.update(x.strip().split())

    max_words = 40
    # print(counter_)

    # # Generate the word cloud
    # wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color='white').generate(text)
    wordcloud = WordCloud(width=800, height=400, max_words=max_words,
                          background_color='white').generate_from_frequencies(counter_)

    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")  # Turn off the axis
    plt.show()

    # Save the word cloud image as a PNG file
    wordcloud.to_file(image_path)


if __name__ == "__main__":
    args = define_arguments()
    # game_version = '{}_{}_{}_{}_{}_runs_{}_rollout_ver_{}_{}'.format(args.commodity, args.seller_instruction, args.buyer_instruction, args.game_type, args.n_exp, args.n_rollout, args.ver, args.moderator_instruction)
    # print('game_version is {}'.format(game_version))

    log_prefix = '{}_{}_{}_{}_{}'.format(args.commodity, args.seller_instruction, args.buyer_instruction, args.remediator_engine, time.strftime("%Y%m%d-%H%M%S"))
    logger = Logger(args.output_path + log_prefix + ".txt", args.verbose)
    print('logger path is {}'.format(args.output_path + log_prefix + ".txt"))

    main(args)
    # image_path = args.output_path + 'images/' + str(args.commodity) + '/' + game_version + "_buyer_wordcloud.png"
    # print_wordcloud(dict_['buyer'], image_path)
    #
    # image_path = args.output_path + 'images/' + str(args.commodity) + '/' + game_version + "_seller_wordcloud.png"
    # print_wordcloud(dict_['seller'], image_path)

