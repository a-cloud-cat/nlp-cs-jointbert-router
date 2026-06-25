import os
import sys
import re
import json
import random
import jieba

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_DIR, INTENT_LABEL_MAP, SLOT_LABEL_MAP, MAX_SEQ_LEN

STOPWORDS = set()
stopwords_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'stopwords.txt')
with open(stopwords_path, 'r', encoding='utf-8') as f:
    for line in f:
        STOPWORDS.add(line.strip())

def clean_text(text):
    text = text.strip()
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def jieba_segment(text):
    words = jieba.lcut(text)
    words = [word for word in words if word not in STOPWORDS and len(word) >= 1]
    return words

def create_smp2019_dataset():
    data = []
    
    templates = {
        '退货申请': [
            {'text': '我想退货', 'slots': []},
            {'text': '申请退货', 'slots': []},
            {'text': '帮我退一下货', 'slots': []},
            {'text': '订单{订单号}想退货', 'slots': ['订单号']},
            {'text': '商品{商品名}要退货', 'slots': ['商品名']},
            {'text': '{商品名}订单{订单号}申请退货', 'slots': ['商品名', '订单号']},
            {'text': '这个{商品名}质量太差了我要退货', 'slots': ['商品名']},
            {'text': '收到的{商品名}有问题想退货', 'slots': ['商品名']},
            {'text': '{商品名}不满意能退吗', 'slots': ['商品名']},
            {'text': '我要退掉{商品名}', 'slots': ['商品名']},
            {'text': '订单{订单号}的{商品名}想退货', 'slots': ['订单号', '商品名']},
            {'text': '买的{商品名}不好用想退', 'slots': ['商品名']},
            {'text': '退货怎么操作', 'slots': []},
            {'text': '如何申请退货', 'slots': []},
            {'text': '退货流程是什么', 'slots': []},
            {'text': '刚收到的{商品名}想退掉', 'slots': ['商品名']},
            {'text': '{商品名}与描述不符想退货', 'slots': ['商品名']},
            {'text': '发错货了想退货', 'slots': []},
            {'text': '商品{商品名}发错了我要退货', 'slots': ['商品名']},
            {'text': '订单{订单号}发错了申请退货', 'slots': ['订单号']},
            {'text': '这个{商品名}有瑕疵', 'slots': ['商品名']},
            {'text': '{商品名}坏了想退货', 'slots': ['商品名']},
            {'text': '质量有问题要求退货', 'slots': []},
            {'text': '{商品名}质量太差劲了', 'slots': ['商品名']},
            {'text': '{商品名}怎么这么差劲', 'slots': ['商品名']},
            {'text': '{商品名}太烂了想退货', 'slots': ['商品名']},
        ],
        '物流查询': [
            {'text': '查一下物流', 'slots': []},
            {'text': '我的快递到哪了', 'slots': []},
            {'text': '物流信息', 'slots': []},
            {'text': '订单{订单号}的物流', 'slots': ['订单号']},
            {'text': '帮我查订单{订单号}到哪了', 'slots': ['订单号']},
            {'text': '{商品名}到哪了', 'slots': ['商品名']},
            {'text': '快递到哪了', 'slots': []},
            {'text': '我的包裹到哪了', 'slots': []},
            {'text': '什么时候能到', 'slots': []},
            {'text': '预计什么时候送达', 'slots': []},
            {'text': '订单{订单号}什么时候到', 'slots': ['订单号']},
            {'text': '帮我追踪一下快递', 'slots': []},
            {'text': '物流单号{订单号}查一下', 'slots': ['订单号']},
            {'text': '快递怎么这么慢', 'slots': []},
            {'text': '物流更新一下', 'slots': []},
            {'text': '查一下订单{订单号}的物流状态', 'slots': ['订单号']},
            {'text': '{商品名}的物流信息', 'slots': ['商品名']},
            {'text': '买的{商品名}到哪了', 'slots': ['商品名']},
            {'text': '怎么还没到', 'slots': []},
            {'text': '快递卡住了吗', 'slots': []},
        ],
        '改地址': [
            {'text': '改一下收货地址', 'slots': []},
            {'text': '地址错了要修改', 'slots': []},
            {'text': '更改收货地址', 'slots': []},
            {'text': '订单{订单号}改地址', 'slots': ['订单号']},
            {'text': '新地址是{收货地址}', 'slots': ['收货地址']},
            {'text': '订单{订单号}改成{收货地址}', 'slots': ['订单号', '收货地址']},
            {'text': '收货地址写错了', 'slots': []},
            {'text': '能改一下地址吗', 'slots': []},
            {'text': '地址要改', 'slots': []},
            {'text': '搬家了地址要改', 'slots': []},
            {'text': '订单{订单号}的地址改一下', 'slots': ['订单号']},
            {'text': '把地址改成{收货地址}', 'slots': ['收货地址']},
            {'text': '我想修改收货地址', 'slots': []},
            {'text': '发货地址错了', 'slots': []},
            {'text': '地址需要变更', 'slots': []},
            {'text': '订单{订单号}地址有误', 'slots': ['订单号']},
            {'text': '送错地方了能改吗', 'slots': []},
            {'text': '新地址是{收货地址}请修改', 'slots': ['收货地址']},
            {'text': '帮我改一下收货地址', 'slots': []},
        ],
        '开发票': [
            {'text': '开发票', 'slots': []},
            {'text': '我要开发票', 'slots': []},
            {'text': '帮忙开发票', 'slots': []},
            {'text': '订单{订单号}开发票', 'slots': ['订单号']},
            {'text': '{商品名}开发票', 'slots': ['商品名']},
            {'text': '发票抬头是{商品名}', 'slots': ['商品名']},
            {'text': '怎么开发票', 'slots': []},
            {'text': '发票怎么开', 'slots': []},
            {'text': '需要开发票', 'slots': []},
            {'text': '请帮我开发票', 'slots': []},
            {'text': '订单{订单号}需要开票', 'slots': ['订单号']},
            {'text': '{商品名}的发票', 'slots': ['商品名']},
            {'text': '电子发票', 'slots': []},
            {'text': '开电子发票', 'slots': []},
            {'text': '增值税发票', 'slots': []},
            {'text': '专票怎么开', 'slots': []},
            {'text': '普票还是专票', 'slots': []},
            {'text': '发票信息', 'slots': []},
            {'text': '开票信息', 'slots': []},
            {'text': '订单{订单号}的发票抬头', 'slots': ['订单号']},
        ],
        '领优惠券': [
            {'text': '领优惠券', 'slots': []},
            {'text': '优惠券在哪领', 'slots': []},
            {'text': '我要领券', 'slots': []},
            {'text': '手机{手机号}领优惠券', 'slots': ['手机号']},
            {'text': '用{手机号}领券', 'slots': ['手机号']},
            {'text': '有优惠券吗', 'slots': []},
            {'text': '优惠码', 'slots': []},
            {'text': '折扣券', 'slots': []},
            {'text': '满减券', 'slots': []},
            {'text': '新人券', 'slots': []},
            {'text': '专属优惠券', 'slots': []},
            {'text': '优惠券怎么领', 'slots': []},
            {'text': '怎么获取优惠券', 'slots': []},
            {'text': '领券入口', 'slots': []},
            {'text': '输入手机号{手机号}领券', 'slots': ['手机号']},
            {'text': '手机号{手机号}可以领券吗', 'slots': ['手机号']},
            {'text': '帮我领优惠券', 'slots': []},
            {'text': '领取优惠', 'slots': []},
            {'text': '优惠活动', 'slots': []},
            {'text': '促销活动', 'slots': []},
        ],
    }
    
    slot_values = {
        '订单号': ['O12345', 'O98765', 'O11111', 'O22222', 'O33333', 'O44444', 'O55555', 'O66666', 'O77777', 'O88888', 'O10000', 'O20000', 'O30000'],
        '手机号': ['13800138000', '13912345678', '15812345678', '18612345678', '13712345678', '13612345678', '18812345678'],
        '商品名': ['手机壳', '蓝牙耳机', '笔记本电脑', '充电宝', '数据线', '鼠标', '键盘', '显示器', '茶杯', '保温杯', '衣服', '鞋子', '包包', '化妆品', '护肤品', '手表', '耳机', '台灯', '书桌', '椅子'],
        '收货地址': ['北京市朝阳区', '上海市浦东新区', '广州市天河区', '深圳市南山区', '杭州市西湖区', '南京市鼓楼区', '成都市锦江区'],
    }
    
    for intent, items in templates.items():
        for item in items:
            for _ in range(10):
                template_text = item['text']
                slot_names = item['slots']
                filled_slots = []
                used_values = set()
                
                for slot_name in slot_names:
                    value = random.choice(slot_values[slot_name])
                    while value in used_values:
                        value = random.choice(slot_values[slot_name])
                    used_values.add(value)
                    template_text = template_text.replace('{' + slot_name + '}', value)
                    filled_slots.append((slot_name, value))
                
                tokens = jieba_segment(template_text)
                
                char_label_map = {}
                for slot_name, slot_value in filled_slots:
                    idx = template_text.find(slot_value)
                    if idx != -1:
                        for i in range(idx, idx + len(slot_value)):
                            if i == idx:
                                char_label_map[i] = f'B-{slot_name}'
                            else:
                                char_label_map[i] = f'I-{slot_name}'
                
                slot_labels = []
                for i, char in enumerate(template_text):
                    slot_labels.append(char_label_map.get(i, 'O'))
                
                data.append({
                    'text': template_text,
                    'intent': intent,
                    'tokens': tokens,
                    'slot_labels': slot_labels,
                    'slots': filled_slots
                })
    
    random.shuffle(data)
    
    split_idx1 = int(len(data) * 0.7)
    split_idx2 = int(len(data) * 0.85)
    
    train_data = data[:split_idx1]
    val_data = data[split_idx1:split_idx2]
    test_data = data[split_idx2:]
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    with open(os.path.join(DATA_DIR, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(DATA_DIR, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(DATA_DIR, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f'Dataset created: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}')
    return train_data, val_data, test_data

def load_dataset(split='train'):
    filepath = os.path.join(DATA_DIR, f'{split}.json')
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    create_smp2019_dataset()
