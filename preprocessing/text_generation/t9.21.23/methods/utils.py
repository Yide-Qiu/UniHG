
import requests
import pdb
import os
import re
import time
# from faker import Faker

# UA = Faker()

# os.environ['NO_PROXY'] = 'stackoverflow.com'

def find_wiki(id):
    time.sleep(0.01)
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&props=labels&ids={id}"
    # headers = {
    #         'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.112 Safari/537.36',
    #     }
    proxies = {
        # 'http': 'http://us.he.07.nekocloud.cn:10002',
        # 'https': 'https://us.he.07.nekocloud.cn:10002',
        # 'http': 'http://127.0.0.1:10809',
        # 'https': 'https://127.0.0.1:10809',
        'http': 'socks5://127.0.0.1:10808',
        'https': 'socks5://127.0.0.1:10808'
        }  

    headers = {
        # 'user-agent' : UA.user_agent(),
        # 'http': '117.86.24.183:40011',     # 键值对形式设置
        # 'https': '106.110.200.153:40023',  # https网站协议对应的IP设置
        'Connection': 'close'
        # 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.56 Safari/537.36'
        }
    print(headers['user-agent'])
    # response = requests.get('https://www.baidu.com', proxies=proxies)
    # print(response.text)
    requests.adapters.DEFAULT_RETRIES = 5
    s = requests.session()
    s.keep_alive = False
    response = requests.get(url, headers=headers, proxies=proxies)

    if response.status_code != 200:
        print("error!")
    else:
        print(response.status_code)

    # response = requests.get(url, verify=False)
    data = response.json()
    label = id
    if 'entities' in data.keys():
        if 'labels' in data["entities"][id].keys():
            label = data["entities"][id]["labels"]["en"]["value"] 
    return label


def entity_or_attribute(ss):
    if ss[0] == ('P' or 'Q') and ss[1:].isdigit() == True:
        return True
    return False


def is_attribute(ss):
    if len(ss) < 2:
        return False
    if ss[0] == 'P' and ss[1:].isdigit() == True:
        return True
    return False

def is_entity(ss):
    if len(ss) < 2:
        return False
    if ss[0] == 'Q' and ss[1:].isdigit() == True:
        return True
    return False

def Bad_Attribute(ss):
    if ss.find('ID') >= 0:
        return True
    
    return False

def entity_select(js):
    # True means be select
    # non_english_pattern = re.compile(r'[^\x00-\x7F]+')
    if 'en' not in js['labels'].keys() or 'en' not in js['descriptions']:
        return False
    if 'P31' not in js['claims'].keys():
        return False
    if 'datavalue' not in js['claims']['P31'][0]['mainsnak'].keys():
        return False
    # if non_english_pattern.search(js['labels']['en']['value']) or non_english_pattern.search(js['descriptions']['en']['value']):
    #     print(js['labels']['en']['value'])
    #     print(js['descriptions']['en']['value'])
    #     pdb.set_trace()
    return True

def label_select(js):
    if 'en' not in js['labels'].keys() or 'en' not in js['descriptions']:
        return False
    return True






