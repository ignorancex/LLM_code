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
from collections import deque

import keys
from agent import (load_initial_instructions, involve_moderator, parse_final_price,
                   BuyerAgent, SellerAgent, ModeratorAgent, SellerCriticAgent, BuyerCriticAgent, RemediatorAgent,
                   load_initial_instructions_withprefix)
from utils import *

from retry.api import retry_call

STOP_AFTER_ATTEMPT = 4

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

CONST_CRITIC_PATH = "lib_prompt/balloon/constant_feedback.txt"
HUMAN_CRITIC_PATH = "lib_prompt/balloon/human_feedback_seller.txt"

dict_ = {'buyer': set(), 'seller': set()}

import argparse


def define_arguments():
    parser = argparse.ArgumentParser()

    # price related to the house
    parser.add_argument('--commodity', type=str, default="industrial",
                        help="[balloon, salary, house, industrial, business]")

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
    parser.add_argument('--seller_instruction', type=str, default="seller_social_norm_explicit_labeling_cn_withgoals")

    parser.add_argument('--reset_seller_initial_history_every_exp', type=bool, default=True,
                        help='generate social norm instruct rules every experiment, used for prefix generation')
    parser.add_argument('--seller_prefix_instruction', type=str, default="seller_social_prefix_cn_simplified_withgoals",
                        help="[seller_social_prefix_cn_simplified, seller_social_prefix_cn, seller_social_prefix_cn_simplified_withgoals]")

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
    parser.add_argument('--game_type', type=str, default='run_ternary',
                        help='[criticize_seller, criticize_buyer, seller_compare_feedback, run_simple, run_ternary]')
    parser.add_argument('--n_exp', type=int, default=100,
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
    parser.add_argument('--remediator_engine', type=str, default="gpt-3.5-turbo")

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
    engine_map = {"seller": SellerAgent,
                  "buyer": BuyerAgent,
                  "seller_critic": SellerCriticAgent,
                  "buyer_critic": BuyerCriticAgent,
                  "moderator": ModeratorAgent,
                  "remediator": RemediatorAgent
                  }

    if ("gpt" in engine_name):
        api_key = args.api_key
    elif ("claude" in engine_name):
        api_key = args.anthropic_api_key
    elif ("j2" in engine_name):
        api_key = args.ai21_api_key
    elif ("cohere" in engine_name):
        api_key = args.cohere_api_key
    else:
        raise ValueError("engine name %s not found" % engine_name)

    engine_class = engine_map[agent_type]

    return engine_class, api_key


def call_engine_for_generating_prefix(prefix_dialog_history, args):
    try:
        completion = retry_call(openai.ChatCompletion.create, fkwargs={"model": "gpt-3.5-turbo",
                                                                       "messages": prefix_dialog_history,
                                                                       "api_key": args.api_key,
                                                                       "n": 1, "temperature": 1.0,
                                                                       "request_timeout": 30}, tries=3, delay=1,
                                jitter=1)
        for choice in completion['choices']:
            # print("====================================")
            chatgpt_message = choice['message']['content']
            print('chatgpt_message is: {}'.format(chatgpt_message))
            return completion['choices'][0]['message']['content']
    except:
        print('-----------------------------openai API is failed!!!!!------------------------------------')
        completion = {'choices': [{'message': {'content': 'Error'}}]}
        return 'null'


def extract_social_norm_rules(res_):
    if "1." not in res_:
        return "1. 例子：使用侮辱性语言 2. 例子：打断别人的发言。 3. 例子：嘲笑或讽刺他人。4. 例子：忽视他人的观点或感受。 5. 例子：过度争论或争执。 "
    else:
        return res_

def run_ternary(args, buyer, seller, moderator, who_is_first="seller", n_exp=5):
    """run multiple experiments without critic, simply checking if the model can play the game
    """
    # print('who_is_first is {}'.format(who_is_first))
    start_time = time.time()
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

    output_lists, prefer_lists = [], []

    for i in range(n_exp):
        logger.write("==== ver %s CASE %d / %d, %.2f min ====" % (args.ver, i, n_exp, compute_time(start_time)))

        if args.reset_seller_initial_history_every_exp is True:
            # seller init
            prefix_dialog_history = load_initial_instructions(
                "lib_prompt/{}/{}.txt".format(args.commodity, args.seller_prefix_instruction))
            res_ = call_engine_for_generating_prefix(prefix_dialog_history, args)
            social_norm_rules = extract_social_norm_rules(res_)

            seller_initial_dialog_history = load_initial_instructions_withprefix(
                'lib_prompt/{}/{}.txt'.format(args.commodity, args.seller_instruction), social_norm_rules)
            seller.reset_initial_dialogue_history(seller_initial_dialog_history)

        buyer.reset()
        seller.reset()
        moderator.reset()

        logger.write('== The seller\'s initial prompt ==')
        logger.write('{}'.format(seller.dialog_history))
        logger.write('== The buyer\'s initial prompt ==')
        logger.write('{}'.format(buyer.dialog_history))

        logger.write('---- start bargaining ----')

        # the first turn of dialogue generated by the seller LLM
        buyer_run = buyer.last_response
        seller_run = seller.call(buyer_run)

        head = {'seller_dialog_history': copy.deepcopy(seller.dialog_history),
                'buyer_dialog_history': copy.deepcopy(buyer.dialog_history),
                'no_deal_cnt': 0,
                'number_of_turns': 1}

        output_list, prefer_list = hierarchical_traversal(head, buyer, seller, moderator, who_is_first=who_is_first, args=args, remediator=remediator)
        output_lists.extend(output_list)
        prefer_lists.extend(prefer_list)
    #     deal_prices.append(price)
    #     if price != -1:
    #         effective_price_num += 1
    #         sum_price += price
    #         logger.write("The final deal price is {}".format(price))
    #     else:
    #         logger.write("The deal is BROKEN!")
    #     logger.write("\n\n\n\n")
    # average_price = 0.0 if effective_price_num == 0 else float(sum_price / effective_price_num)
    # logger.write("\nSuccessful deal rate: {}%, average price: {}".format(float(effective_price_num / n_exp * 100.0),
    #                                                                      average_price))
    # logger.write("\nDeal prices are: {}".format(' '.join([str(x) for x in deal_prices]).strip()))
    return output_lists, prefer_lists

def hierarchical_traversal(head, buyer, seller, moderator, who_is_first, args, remediator=None, no_deal_thres=10, n_round=10):
    # conversation trajectories;
    output_list = []
    # preference instances;
    prefer_list = []
    # Create an empty deque
    queue = deque()
    queue.append(head)

    start_involve_moderator = False
    deal_at = "none"

    # # todo only first sentence has violation, using COT?
    #
    # if (who_is_first == "buyer"):
    #     logger.write('  buyer: %s' % buyer.last_response)
    #     dict_['buyer'].add(buyer.last_response)
    #     seller_run = seller.last_response
    #     buyer_run = buyer.call(seller_run)
    #     if remediator is not None and ('【violation】' in buyer.dialog_history[-1]['content'] or
    #                                    '【违规】' in buyer.dialog_history[-1]['content'] or
    #                                    '【违反社会规范】' in buyer.dialog_history[-1]['content'] or
    #                                    '【违反规范】' in buyer.dialog_history[-1]['content'] or
    #                                    'violation' in buyer.dialog_history[-1]['content'] or
    #                                    '违规' in buyer.dialog_history[-1]['content']):
    #         logger.write('【Before remediation】: %s' % buyer.dialog_history[-1]['content'])
    #         sentence = remediator.produce_remediation(buyer.dialog_history)
    #         buyer.remediate_conversation(sentence)
    #         logger.write('【After remediation】: %s' % buyer.dialog_history[-1]['content'])
    #         buyer_run = sentence

    logger.write('  seller: %s' % seller.last_response)
    logger.write('  buyer: %s' % buyer.last_response)

    while queue:
        nodeCount = len(queue)
        while nodeCount > 0:

            node_dict = queue.popleft()

            nodeCount -= 1

            seller.reset_history(node_dict['seller_dialog_history'])
            buyer.reset_history(node_dict['buyer_dialog_history'])
            layer_no_deal_cnt = node_dict['no_deal_cnt']
            number_of_turns = node_dict['number_of_turns']

            deal_at = "none"

            # ------------- judge whether to end the conversation---------------
            seller_run = seller.last_response
            buyer_run = buyer.last_response
            start_involve_moderator = False

            if (start_involve_moderator is False and involve_moderator(buyer_run, seller_run)):
                start_involve_moderator = True
                logger.write('---- start moderating ----')

            end_flag = False
            if (start_involve_moderator):
                moderate = moderator.moderate(seller.dialog_history, who_was_last="seller")
                logger.write('MODERATE have the seller and the buyer achieved a deal? Yes or No: %s' % moderate)
                if ("yes" in moderate.lower()):
                    deal_at = "seller"
                    end_flag = True
                else:
                    no_deal_cnt = layer_no_deal_cnt + 1
                    if (no_deal_cnt == no_deal_thres) or (number_of_turns == n_round):
                        end_flag = True

            if end_flag is True:
                if (deal_at != "none"):
                    if (deal_at == "seller"):
                        final_price = parse_final_price(seller.dialog_history)
                    else:
                        final_price = parse_final_price(buyer.dialog_history)
                else:
                    final_price = -1
                output_list.append({'final_price': final_price,
                                    'seller_dialog_history': copy.deepcopy(node_dict['seller_dialog_history']),
                                    'buyer_dialog_history': copy.deepcopy(node_dict['buyer_dialog_history'])})

            # --------------------------------------------------------------------
            else:
                # --------------------- Generate branches ------------------------
                # original sentence
                end_flag = False
                buyer_run = buyer.call(seller_run)

                if (start_involve_moderator):
                    moderate = moderator.moderate(buyer.dialog_history, who_was_last="buyer")
                    logger.write('MODERATE have the seller and the buyer achieved a deal? Yes or No: %s' % moderate)
                    if ("yes" in moderate.lower()):
                        deal_at = "buyer"
                        end_flag = True
                    else:
                        no_deal_cnt = layer_no_deal_cnt + 2
                        if (no_deal_cnt == no_deal_thres) or (number_of_turns == n_round):
                            end_flag = True

                if end_flag is True:
                    if (deal_at != "none"):
                        if (deal_at == "seller"):
                            final_price = parse_final_price(seller.dialog_history)
                        else:
                            final_price = parse_final_price(buyer.dialog_history)
                    else:
                        final_price = -1
                    output_list.append({'final_price': final_price,
                                        'seller_dialog_history': copy.deepcopy(node_dict['seller_dialog_history']),
                                        'buyer_dialog_history': copy.deepcopy(node_dict['buyer_dialog_history'])})
                else:
                    seller_run = seller.call(buyer_run)

                    # original new seller sentence
                    temporal_history = copy.deepcopy(seller.dialog_history)
                    # temporal_history[-1]['content'] = temporal_history[-1]['content'].replace("【violation】", "").replace("violation", "")
                    new_node_dict = {'seller_dialog_history': temporal_history,
                                     'buyer_dialog_history': copy.deepcopy(buyer.dialog_history),
                                     'no_deal_cnt': layer_no_deal_cnt + 1,
                                     'number_of_turns': number_of_turns + 1}
                    queue.append(new_node_dict)

                    # remediation sentence
                    if remediator is not None and ('【violation】' in seller.dialog_history[-1]['content'] or
                                                   '【违规】' in seller.dialog_history[-1]['content'] or
                                                   '【违反社会规范】' in seller.dialog_history[-1]['content'] or
                                                   '【违反规范】' in seller.dialog_history[-1]['content'] or
                                                   'violation' in seller.dialog_history[-1]['content'] or
                                                   '违规' in seller.dialog_history[-1]['content']):

                        # end_flag = False

                        # prefer_list.append(node_dict)

                        seller.reset_history(node_dict['seller_dialog_history'])
                        buyer.reset_history(node_dict['buyer_dialog_history'])

                        logger.write('【Before remediation】: %s' % seller.dialog_history[-1]['content'])
                        sentence = remediator.produce_remediation(seller.dialog_history)
                        logger.write('【Remediation】: %s' % sentence)

                        prefer_list.append({'seller_dialog_history': copy.deepcopy(node_dict['seller_dialog_history']),
                                            'buyer_dialog_history': copy.deepcopy(node_dict['buyer_dialog_history']),
                                            'remediation': sentence})

                        seller.remediate_conversation(sentence)
                        new_node_dict = {'seller_dialog_history': copy.deepcopy(seller.dialog_history),
                                             'buyer_dialog_history': copy.deepcopy(buyer.dialog_history),
                                             'no_deal_cnt': layer_no_deal_cnt + 1,
                                             'number_of_turns': number_of_turns + 1}
                        queue.append(new_node_dict)

                        # seller_run = sentence
                        # buyer_run = buyer.call(seller_run)
                        #
                        # if (start_involve_moderator):
                        #     moderate = moderator.moderate(buyer.dialog_history, who_was_last="buyer")
                        #     logger.write('MODERATE have the seller and the buyer achieved a deal? Yes or No: %s' % moderate)
                        #     if ("yes" in moderate.lower()):
                        #         deal_at = "buyer"
                        #         end_flag = True
                        #     else:
                        #         no_deal_cnt = layer_no_deal_cnt + 2
                        #         if (no_deal_cnt == no_deal_thres) or (number_of_turns == n_round):
                        #             end_flag = True
                        #
                        # if end_flag is True:
                        #     if (deal_at != "none"):
                        #         if (deal_at == "seller"):
                        #             final_price = parse_final_price(seller.dialog_history)
                        #         else:
                        #             final_price = parse_final_price(buyer.dialog_history)
                        #     else:
                        #         return -1
                        #     output_list.append({'final_price': final_price,
                        #                         'seller_dialog_history': copy.deepcopy(node_dict['seller_dialog_history']),
                        #                         'buyer_dialog_history': copy.deepcopy(node_dict['buyer_dialog_history'])})
                        # else:
                        #     seller_run = seller.call(buyer_run)
                        #
                        #     new_node_dict = {'seller_dialog_history': copy.deepcopy(seller.dialog_history),
                        #                      'buyer_dialog_history': copy.deepcopy(buyer.dialog_history),
                        #                      'no_deal_cnt': layer_no_deal_cnt + 1,
                        #                      'number_of_turns': number_of_turns + 1}
                        #     queue.append(new_node_dict)

    return output_list, prefer_list


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
    buyer_initial_dialog_history = load_initial_instructions(
        'lib_prompt/{}/{}.txt'.format(args.commodity, args.buyer_instruction))
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
    moderator_initial_dialog_history = load_initial_instructions(
        "lib_prompt/{}/{}.txt".format(args.commodity, args.moderator_instruction))
    moderator_engine_class, moderator_api_key = get_engine_and_api_key(engine_name=args.moderator_engine,
                                                                       agent_type="moderator", args=args
                                                                       )
    moderator = moderator_engine_class(initial_dialog_history=moderator_initial_dialog_history,
                                       agent_type="moderator", engine=args.moderator_engine, api_key=moderator_api_key,
                                       trace_n_history=args.moderator_trace_n_history
                                       )

    if (args.game_type in ['criticize_seller', 'criticize_buyer', 'seller_compare_feedback', 'run_simple', 'run_ternary']):
        pass
    else:
        raise ValueError(
            "game_type must be in ['criticize_seller', 'criticize_buyer', 'seller_compare_feedback', 'run_simple', 'run_ternary']")

    # run
    if args.commodity == "salary":
        who_is_first = "buyer"
    else:
        who_is_first = "seller"
    if (args.buyer_instruction == "buyer_no_initial_price"):
        who_is_first = "buyer"
    if (args.game_type == "run_ternary"):
        output_lists, prefer_lists = run_ternary(args, buyer, seller, moderator, n_exp=args.n_exp, who_is_first=who_is_first)
        log_prefix = '{}_{}_{}_{}_{}'.format(args.game_type, args.commodity, args.seller_instruction,
                                             args.buyer_instruction,
                                             time.strftime("%Y%m%d-%H%M%S"))
        output_lists_path = args.output_path + log_prefix + "_output_lists.json"
        prefer_lists_path = args.output_path + log_prefix + "_prefer_lists.json"
        # Write the list to a JSON file with indentation (e.g., 4 spaces)
        with open(output_lists_path, 'w', encoding="utf-8") as json_file:
            json.dump(output_lists, json_file, indent=4, ensure_ascii=False)
            logger.write('write into {}'.format(output_lists_path))
        with open(prefer_lists_path, 'w', encoding="utf-8") as json_file:
            json.dump(prefer_lists, json_file, indent=4, ensure_ascii=False)
            logger.write('write into {}'.format(prefer_lists_path))
    return


if __name__ == "__main__":
    args = define_arguments()
    # game_version = '{}_{}_{}_{}_{}_runs_{}_rollout_ver_{}_{}'.format(args.commodity, args.seller_instruction, args.buyer_instruction, args.game_type, args.n_exp, args.n_rollout, args.ver, args.moderator_instruction)
    # print('game_version is {}'.format(game_version))

    log_prefix = '{}_{}_{}_{}_{}'.format(args.game_type, args.commodity, args.seller_instruction, args.buyer_instruction,
                                      time.strftime("%Y%m%d-%H%M%S"))
    logger = Logger(args.output_path + log_prefix + ".txt", args.verbose)
    print('logger path is {}'.format(args.output_path + log_prefix + ".txt"))

    main(args)
    # image_path = args.output_path + 'images/' + str(args.commodity) + '/' + game_version + "_buyer_wordcloud.png"
    # print_wordcloud(dict_['buyer'], image_path)
    #
    # image_path = args.output_path + 'images/' + str(args.commodity) + '/' + game_version + "_seller_wordcloud.png"
    # print_wordcloud(dict_['seller'], image_path)

