import json
import random

random.seed(1987)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# # Create a Counter object
# counter = Counter()
#
# list_ = ['I can\'t go that low, but I can offer it to you for $18.', 'I understand your offer, but the lowest I can go is $17.', 'Great! Deal! The balloon is yours for $17. Enjoy!', 'Hi, this is a good balloon and its price is $20']
# for x in list_:
#     counter.update(x.strip().split())
#
# # # Convert the list of words into a single string
# # text = " ".join(word_list)
#
# max_words = 30
# print(counter)
#
# # # Generate the word cloud
# # wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color='white').generate(text)
# wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color='white').generate_from_frequencies(counter)
#
# # Display the word cloud using matplotlib
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")  # Turn off the axis
# plt.show()
#
# # Save the word cloud image as a PNG file
# wordcloud.to_file("wordcloud.png")
#
# # list_ = [1,2,3,4,6]
# # string_ = ' '.join([str(x) for x in list_]).strip()
# # print(string_)


def extract():
    content = ""
    with open('test.txt', 'r') as file:
        # Read all lines
        lines = file.readlines()
    for line in lines:
        content += line
    content = content.strip()
    lists = [x.strip() for x in content.split('\n') if x.strip()!='']
    print(lists)

    level_2_result, level_1_result = "", ""
    for i, item in enumerate(lists):
        if '2. conversion' in item.lower() and i < len(lists)-1:
            level_2_result = lists[i+1]
        if 'further conversion' in item.lower() and i < len(lists)-1:
            level_1_result = lists[i+1]
    return level_2_result, level_1_result


def main():
    # print(extract())
    str_ = " <s> System: 你是一个对话助手，你的任务是改写对话中最后一句话，使其符合社会规范。\n在谈判游戏中，谈判的对话如下：\n人物A: 你好，请问贵公司有一种特殊的工业商品吗？\n人物B：你好，我们公司产能充足，能够以一次性的方式向贵司出售所需的工业商品。这种工业商品的单元价格是$50。\n人物A: 你会考虑以每单元$30的价格进行出售吗？\n人物B：很抱歉，我们不会接受低于$40的价格。\n在这个对话中，最后一句话\"很抱歉，我们不会接受低于$40的价格\"违反了社会规范，请对这句话进行改写:\n</s><s> Assistant: 很遗憾，我们的起始报价是您无法满足的需求价码。\n</s>"
    list_ = str_.split('<s>')
    print(list_)
    remedy_ = str_.split('<s>')[-1].replace('</s>', '').replace('Assistant:', '').replace('assistant:', '').strip()
    print(remedy_)

def toss_coin(prob_):
    # Generating a random number between 0 and 1
    result = random.random()

    # Assigning heads or tails based on probabilities
    if result < prob_:
        return 'Heads'
    else:
        return 'Tails'

def toss():
    prob_ = 1.0
    # Tossing the coin 10 times for demonstration
    for _ in range(10):
        outcome = toss_coin(prob_)
        print(outcome)

def find_answer_index(my_list):
    for index, element in enumerate(my_list):
        if isinstance(element, str) and element.startswith('answer:'):
            return index
    return None  # Return None if no such element is found

def calc_length(dict_):
    dict_json_str = json.dumps(dict_, ensure_ascii=False)
    length = len(dict_json_str)
    return(length)

def assemble(contents_list):
    str_ = ''
    for list_ in contents_list:
        str_ += '\n'.join(list_) + '\n\n'
    return str_.strip()

def disassemble(dict_):
    contents_list = []
    contents = [x for x in dict_['content'].split('\n') if x != '']
    contents_list.append([contents[0]])
    contents = contents[1:]
    while True:
        index = find_answer_index(contents)
        if index:
            contents_list.append(contents[0:index+1])
            contents = contents[index+1:]
        else:
            break
    return contents_list

def reduce_contents_list(contents_list):
    # Ensure the list has more than two elements
    if len(contents_list) <= 2:
        return contents_list

    # Generate a random index that is not the first or last
    index = random.randint(1, len(contents_list) - 2)

    # Remove the element at the random index
    contents_list.pop(index)

    return contents_list

def clip_moderate(dict_):
    while calc_length(dict_) >= 4000:
        contents_list = disassemble(dict_)
        contents_list = reduce_contents_list(contents_list)
        dict_['content'] = assemble(contents_list)
    return dict_


def test():
    res_ = []
    for _ in range(5):
        n_round = 10
        no_deal_cnt = 0
        no_deal_thres = 5
        deal_at = "none"
        for _ in range(n_round):
            violate_signal = random.random()

            start_involve_moderator = True

            if start_involve_moderator:
                if violate_signal < 0.4:
                    if violate_signal < 0.3:
                        deal_at = "seller"
                    break
                else:
                    no_deal_cnt += 1
                    if no_deal_cnt == no_deal_thres:
                        break

            violate_signal = random.random()

            if (start_involve_moderator):
                if violate_signal < 0.4:
                    if violate_signal < 0.3:
                        deal_at = "buyer"
                    break
                else:
                    no_deal_cnt += 1
                    if no_deal_cnt == no_deal_thres:
                        break

        if (deal_at != "none"):
            if (deal_at == "seller"):
                print('deal_at is seller!')
                res_.append(1)
            else:
                print('deal_at is buyer!')
                res_.append(2)
        else:
            print('deal_at is none!')
            res_.append(-1)
    return res_

def reorder():
    # Sample dictionary
    my_dict = {'apple': 3, 'banana': 1, 'cherry': 2}

    # Sorting the dictionary by values in ascending order
    sorted_dict = sorted(my_dict.items(), key=lambda item: item[1], reverse=True)
    print(sorted_dict)



if __name__ == "__main__":
    # main()
    # toss()
    # dict_ = {'role': 'assistant', 'content': '是的，我准备好玩游戏了！'}
    # calc_length(dict_)

    # dict_ = {'role': 'user', 'content': "Let's play a game. You are the moderator of a bargaining game. Your role is to decide if a seller and a buyer have reached a deal during the bargaining following the example:\n\nseller: 很抱歉，每单位$30的价格无法满足我们的最低保留价格。我们的起始价格是$50。\nbuyer: 感谢您提供的信息，我们会考虑您的起始价格。请问您可否告诉我们有关该工业商品的更多细节和特性？这将帮助我们更好地评估其价值。\nseller: 当然，这款工业商品是由高品质材料制成的，具有优异的耐用性和可靠性。它还具有高效的工作性能和灵活的适应性，可以满足各种工业需求。此外，我们还提供售后服务和技术支持，以确保您在使用过程中得到全面的支持。这些特性使得该工业商品成为您企业的理想选择。\nbuyer: 非常感谢您提供的详细信息。这些特性确实符合我们对工业商品的需求。考虑到您所提供的高品质和优势特点，我们可以在单元价格上适当加强竞争力。我们愿意以每单位$40的价格进行购买，这会对我们双方都带来更大的经济利益。您认为这个价格合理吗？\nquestion: have the seller and the buyer achieved a deal? Yes or No\nanswer: Because both the buyer and seller were still negotiating and haven't agreed on a deal price. The answer is No.\n\nbuyer: 除了价格，我们是否可以考虑增加订单量或延长合同期限以达成更有利的交易条件呢？\nseller: 当然，我们可以考虑增加订单量或延长合同期限来寻求更有利的交易条件。请告诉我您期望的订单量和合同期限，以便我们进一步讨论。\nbuyer: 我们希望能够增加订单量至每次购买1000单元，并将合同期限延长至12个月。这样的安排是否合适呢？\nquestion: have the seller and the buyer achieved a deal? Yes or No\nanswer: Because both the buyer and seller were still negotiating and haven't agreed on a deal price. The answer is No.\n\nbuyer: 我了解你的立场，也理解你坚持的保留价格。考虑到双方的利益，我愿意接受每单位$49的价格。这是我能够提供的最后优惠，请考虑一下。\nseller: 谢谢你的理解和努力。考虑到你的提议和我们双方的利益，我同意以每单位$49的价格进行交易。我相信这将是一个公平的协议，我们可以共同受益。谢谢你的合作！\nbuyer: 非常感谢你的合作和开放的协商态度！我也相信这个协议对双方都是有利的。我们将以每单位$49的价格进行交易。愿我们的合作取得成功！再次感谢你的合作！\nquestion: have the seller and the buyer achieved a deal? Yes or No\nanswer: Because both parties have confirmed the price to be $49. The answer is Yes.\n\nseller: 非常感谢您的理解和合作。我会认真考虑您的提议，并在尽快的时间内给予您一个明确的回复。如果有任何需要进一步讨论的问题或要求，我会及时与您联系。再次感谢您的兴趣和耐心等待。\nbuyer: 非常感谢！期待收到您的回复。如有任何问题或需要进一步的帮助，请随时告知。祝您一切顺利！\nseller: 非常感谢您的好意！我会尽快回复您并提供相关信息。如果您有任何问题或需要进一步的帮助，请随时与我联系。祝您一切顺利！\nquestion: have the seller and the buyer achieved a deal? Yes or No\nanswer: Because the seller would respond with the buyer later and end the conversation. The answer is Yes.\n\nseller: 我明白你们希望以更低的价格购买，但是以每单位$20的价格是低于我们可接受的范围的。我们希望能够在每单位$30左右的价格范围内谈判。\nbuyer: 我理解你们的底线是每单位$30。让我们探索一些其他的方式来满足双方的利益。有没有其他的合作方案或者折扣政策能够在单价上进行灵活调整？\nseller: 我们非常愿意探索其他的合作方案和折扣政策，以满足双方的利益。请告诉我您希望考虑的具体方案或者要求，我们可以讨论并寻找更多的共赢机会。\nquestion: have the seller and the buyer achieved a deal? Yes or No\nanswer: Because both the buyer and seller were still negotiating and haven't agreed on a deal price. The answer is No.\n\nbuyer: 非常感谢您的美好祝福和支持。我也相信在我们共同的努力下，我们能够取得共同的成功。祝愿您一切顺利！再次感谢您的合作！谢谢！\nseller: 非常感谢您的美好祝福和支持。我也坚信在我们的共同努力下，我们一定能够取得共同的成功。祝愿您一切顺利！再次感谢您的合作！谢谢！期待未来的合作！\nbuyer: 非常感谢您的美好祝福和支持。我也坚信在我们的共同努力下，我们一定能够取得共同的成功。祝福您一切顺利！再次感谢您的合作！期待未来的合作！谢谢！\nquestion: have the seller and the buyer achieved a deal? Yes or No\nanswer: Because both the buyer and seller broke the deal. The answer is Yes.\n\nseller: 非常感谢您的合作和积极的态度。我也期待能够安排一个会议，以更深入地讨论订单量和价格的细节，并为共同达成一个长期合作协议铺平道路。我们将通过会议进一步加深我们的关系，并确保双方的利益最大化。请告诉我您的时间偏好，我们会尽快安排一个会议。\nbuyer: 非常感谢您的合作和积极的态度。我们可以在下周四的上午10点进行会议，以便就订单量和价格的细节进行深入讨论。请问这个时间对您来说是否合适？\nseller: 非常感谢您的建议。下周四上午10点的时间对我来说是合适的。我们可以将会议安排在我们的办公室，或者通过视频会议进行，以便更加方便和高效地进行讨论。请提供您的偏好，以便我们确定会议的具体安排。\nquestion: have the seller and the buyer achieved a deal? Yes or No\nanswer: Because both the parties would arrange another meeting to further discuss about the details and end the  current conversation. The answer is Yes.\n\n\nbuyer: 我明白您的限制和最后底线。考虑到双方的需求，我可以接受每单元$38的价格。我们能在这个基础上达成交易吗？\nseller: 非常感谢您的理解和合作。基于双方的谈判，我们可以接受每单元$38的价格。我们可以达成交易。谢谢！\nbuyer: 非常感谢您的合作和让步。我也很高兴能达成交易。我们将尽快安排付款和发货事宜。期待与您建立更加牢固的合作关系！谢谢！\nquestion: have the seller and the buyer achieved a deal? Yes or No\nanswer: Because both parties have confirmed the price to be $38. The answer is Yes.\n\nbuyer: 我理解您对经济利益最大化的追求，我会将您提出的建议转达给我们的团队，看是否能够考虑以每单位$40的价格进行交易。我会尽力协调双方利益，实现一个共赢的结果。\nseller: 非常感谢您的理解和努力。建立共赢的结果是我们共同的目标，我期待我们能达成一个互惠互利的交易。如果您有任何进一步的讨论或建议，请随时告诉我。\nuyer: 非常感谢您的合作和支持。我们会继续努力探讨可能的解决方案，以达成一个互惠互利的交易。如果有任何进一步的讨论或建议，我会及时与您联系。希望我们能够建立起更加牢固的合作关系。\nquestion: have the seller and the buyer achieved a deal? Yes or No\nanswer: Because the seller would respond with the buyer later and end the conversation. The answer is Yes.\n\nseller: 非常抱歉，$30的价格无法满足我们公司的预期。我们可以考虑以更合理的价格进行交易。\nbuyer: 理解。那么，请问你能否给出一个更具竞争力的价格？我相信我们可以找到一个互惠互利的解决方案。\nseller: 我能理解您希望获得更具竞争力的价格，但我们的最低价格是每单位$40。这是我们公司的底线价格，如果您能接受，我们可以进一步讨论合作细节。\nquestion: have the seller and the buyer achieved a deal? Yes or No\nanswer: Because the seller did not provide a clear response to the buyer's inquiry about whether they would accept the proposed transaction price. The answer is No\n\nseller: 非常感谢您的提议。我理解您对价格的关注，但考虑到供应链成本和市场价值，我们最低只能接受$40的价格。这是我们的底线。\nbuyer: 我明白您的顾虑，但我们仍然认为$40的价格对于我们来说过高。我们能否达成一致，以$38的价格完成交易？这样对双方都有利。\nquestion: have the seller and the buyer achieved a deal? Yes or No\nanswer: Because the seller did not provide a clear response to the buyer's inquiry about whether they would accept the proposed transaction price. The answer is No\n\nseller: 我明白你们的限制，但是我们的底线价格是$50每单位。我们的产品具有高品质和价值，与竞争对手相比具有优势。我可以考虑将价格调整到$45每单位的水平，这是我们所能接受的最低限度。能否考虑这个价格呢？\nbuyer: 我理解你们产品的价值和竞争优势，但对于我们来说，$45每单位还是超出了我们的预算范围。我们的底线价格是$40每单位。能否考虑将价格调整到这个水平以满足双方的利益呢？\nquestion: have the seller and the buyer achieved a deal? Yes or No\nanswer: Because the seller did not provide a clear response to the buyer's inquiry about whether they would accept the proposed transaction price. The answer is No\n\nbuyer: 我了解您的立场，但由于我们的预算有限，每单元$35是我们的极限报价。如果你能够接受这个价格，我们可以尽快达成交易并建立更加紧密的合作关系。\nseller: 好的，考虑到我们之间建立更紧密的合作关系的重要性，我愿意接受每单元$35的价格。我们可以立即达成交易并确立更强大的合作关系。感谢您提出的优惠报价。\nbuyer: 非常感谢您的配合和信任。我相信通过这次交易，我们的合作关系将得到进一步巩固和发展。我会尽快与您协调订单细节，并确保交货的及时性。期待与您的持续合作。\nquestion: have the seller and the buyer achieved a deal? Yes or No\nanswer: Because both parties have confirmed the price to be $35. The answer is Yes.\n\nbuyer: 非常感谢您的合作和耐心！我也非常期待与您进一步沟通并讨论订单数量和长期合作协议的细节。我们可以通过会议或邮件进行进一步的交流。我相信通过双方的合作努力，我们一定能达成一个双方满意的协议。再次感谢您的合作，期待与您的进一步沟通！\nseller: 非常感谢您的积极配合和耐心！我也坚信通过我们的合作努力，我们一定能够达成一个双方满意的协议。我期待着与您在会议或邮件中进行更深入的讨论，以确保我们达成一个互惠互利的合作协议。再次感谢您的合作，期待与您的进一步沟通！祝您有愉快的一天！\nanswer: Because both the parties would discuss further in the future and end the conversation now. The answer is Yes.\n\nbuyer: 你好，请问贵公司有一种特殊的工业商品吗？\nseller: 你好，我们公司产能充足，能够以一次性的方式向贵司出售所需的工业商品。这种工业商品的单元价格是$50。\nbuyer: 你会考虑以每单元$30的价格进行出售吗？\nseller: 谢谢您的提议，但根据我们的定价策略，我们目前的起始价格是$50每单位。\nbuyer: 我理解您的定价策略，但考虑到我们是长期的潜在合作伙伴，是否可以稍微优惠一些，以便我们可以达成更有利的交易条件？\nquestion: have the seller and the buyer achieved a deal? Yes or No\nanswer:"}
    # print(clip_moderate(dict_))

    # print(test())
    reorder()



