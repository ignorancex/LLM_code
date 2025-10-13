from collections import defaultdict as df

import requests
from lxml import html


# get crawled list
def get_crawler_list(url):
    response = requests.get(url)
    if response.status_code != 200:
        print("Crawler Failed...")
        return
    tree = html.fromstring(response.content)
    crawler_list = tree.xpath(
        '//*[@id="pytorch-left-menu"]/div/div/ul[4]//li[contains(@class, "toctree-l1")]//@href'
    )
    return crawler_list


# get content with specific labels
def crawler(main_url, model_url):
    url = main_url + model_url
    model_name = model_url.split(".html")[0]
    if "torch" not in model_name:
        model_name = "torch." + model_name

    response = requests.get(url)
    if response.status_code != 200:
        print("Crawler Failed...")
        return

    tree = html.fromstring(response.content)
    apis = tree.xpath(
        '//code[contains(@class, "xref py py-obj docutils literal notranslate")]/parent::*/@title'
    )
    for api in apis:
        torch_dic[model_name].append(api)


# save res
def save_result(torch_dic, total_filename):
    with open(total_filename, "w") as f_total:
        for modle_name, api_lst in torch_dic.items():
            with open(f"{modle_name}.txt", "w") as f:
                for api in api_lst:
                    f.write(api + "\n")
                    f_total.write(api + "\n")


if __name__ == "__main__":
    main_url = "https://pytorch.org/docs/main/"
    torch_dic = df(list)
    # crawler_list = ["torch.html", "nn.html", "nn.functional.html"]
    crawler_list = ["linalg.html"]
    for crawler_name in crawler_list:
        crawler(main_url, crawler_name)

    save_result(torch_dic, "torch_linalg.txt")

    """
    get urls of all the modules
    """
    # crawler_list = get_crawler_list("https://pytorch.org/docs/main/")

    # for x in crawler_list:
    #     print(x)
