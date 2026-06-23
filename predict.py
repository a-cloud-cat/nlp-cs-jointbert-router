import torch
import os
import re
from transformers import BertTokenizer
from config import *
from model import JointBERT

COMMON_PRODUCTS = ['手机壳', '蓝牙耳机', '笔记本电脑', '充电宝', '数据线', '鼠标', '键盘', '显示器', '茶杯', '保温杯', '衣服', '鞋子', '包包', '化妆品', '护肤品', '手表', '耳机', '台灯', '书桌', '椅子', '电脑', '手机', '平板', '音箱', '路由器', '硬盘', '充电器', '电池', 'U盘', '摄像头', '耳机套']

INTENT_KEYWORDS = {
    '退货申请': ['退货', '退款', '退掉', '不满意', '质量差', '差劲', '烂', '坏了', '有问题', '瑕疵', '发错', '想退', '要退', '申请退货'],
    '物流查询': ['物流', '快递', '到哪了', '包裹', '送达', '什么时候到', '快递到了吗', '查物流', '追踪', '派送'],
    '改地址': ['改地址', '地址错了', '搬家', '变更地址', '换地址', '地址要改'],
    '开发票': ['发票', '开票', '电子发票', '专票', '普票', '增值税'],
    '领优惠券': ['优惠券', '领券', '折扣', '满减', '新人券', '优惠码'],
    '转人工': ['转人工', '人工客服', '找人工', '人工服务']
}

class Predictor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.model = JointBERT()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pt'), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def extract_slots_by_rules(self, text):
        slots = {}
        
        order_pattern = r'(订单号?|订单)?[：: ]?([A-Za-z][0-9]{5,8}|[0-9]{8,12})'
        order_match = re.search(order_pattern, text)
        if order_match:
            slots['订单号'] = order_match.group(2)
        
        phone_pattern = r'1[3-9]\d{9}'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            slots['手机号'] = phone_match.group(0)
        
        address_keywords = ['北京市', '上海市', '广州市', '深圳市', '杭州市', '南京市', '成都市', '武汉市', '重庆市', '天津市', '苏州市', '西安市', '长沙市', '郑州市', '东莞市', '佛山市', '厦门市', '沈阳市', '青岛市', '宁波市']
        for addr in address_keywords:
            if addr in text:
                start_idx = text.find(addr)
                end_idx = start_idx + len(addr)
                remaining = text[end_idx:]
                district_match = re.match(r'([\u4e00-\u9fa5]{2,6})', remaining)
                if district_match:
                    slots['收货地址'] = addr + district_match.group(1)
                else:
                    slots['收货地址'] = addr
                break
        
        return slots
    
    def extract_product(self, text):
        for product in COMMON_PRODUCTS:
            if product in text:
                return product
        
        product_patterns = [
            r'(这个|那个|买的|收到的|商品)\s*([\u4e00-\u9fa5]{2,6})\s*(质量|太差|有问题|坏了|不满意|想退|退货)',
            r'([\u4e00-\u9fa5]{2,6})\s*(要退货|想退货|申请退货|能退吗)',
            r'([\u4e00-\u9fa5]{2,6})\s*(开发票|的发票)',
            r'([\u4e00-\u9fa5]{2,6})\s*(物流|快递|到哪了)',
        ]
        
        for pattern in product_patterns:
            match = re.search(pattern, text)
            if match:
                product = match.group(2) if len(match.groups()) > 1 else match.group(1)
                if product not in ['退货', '物流', '发票', '优惠券', '地址', '订单', '手机', '太差', '差劲', '问题', '质量', '商品']:
                    return product
        
        return None
    
    def detect_intent_by_rules(self, text):
        for intent, keywords in INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return intent
        return None
    
    def predict(self, text):
        text = text.strip()
        if not text:
            return {'intent': 'default', 'slots': {}}
        
        rule_intent = self.detect_intent_by_rules(text)
        
        encoding = self.tokenizer(
            text,
            max_length=MAX_SEQ_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            intent_logits, slot_preds = self.model(input_ids, attention_mask)
        
        intent_probs = torch.softmax(intent_logits, dim=1)
        max_prob, intent_idx = torch.max(intent_probs, dim=1)
        model_intent = INTENT_LABELS[intent_idx]
        
        if rule_intent and max_prob.item() < 0.7:
            final_intent = rule_intent
        else:
            final_intent = model_intent
        
        rule_slots = self.extract_slots_by_rules(text)
        
        product_name = None
        if final_intent in ['退货申请', '开发票', '物流查询']:
            product_name = self.extract_product(text)
        
        slot_tags = []
        if slot_preds:
            for idx in slot_preds[0]:
                slot_tags.append(SLOT_IDX_MAP.get(idx, 'O'))
        
        slot_results = {}
        current_slot = None
        current_chars = []
        
        for i, tag in enumerate(slot_tags):
            if tag.startswith('B-'):
                if current_slot:
                    value = ''.join(current_chars).strip()
                    if value and len(value) >= 2:
                        slot_results[current_slot] = value
                current_slot = tag[2:]
                current_chars = [text[i] if i < len(text) else '']
            elif tag.startswith('I-'):
                if current_slot:
                    current_chars.append(text[i] if i < len(text) else '')
            else:
                if current_slot:
                    value = ''.join(current_chars).strip()
                    if value and len(value) >= 2:
                        slot_results[current_slot] = value
                current_slot = None
                current_chars = []
        
        if current_slot:
            value = ''.join(current_chars).strip()
            if value and len(value) >= 2:
                slot_results[current_slot] = value
        
        final_slots = {}
        for slot_name in ['订单号', '手机号', '商品名', '收货地址']:
            if slot_name == '商品名' and product_name:
                final_slots['商品名'] = product_name
            elif slot_name in rule_slots:
                final_slots[slot_name] = rule_slots[slot_name]
            elif slot_name in slot_results:
                val = slot_results[slot_name]
                if not re.search(r'(退货|物流|发票|优惠券|地址)', val):
                    final_slots[slot_name] = val
        
        return {
            'intent': final_intent,
            'slots': final_slots,
            'confidence': max_prob.item()
        }

if __name__ == '__main__':
    predictor = Predictor()
    
    test_cases = [
        '订单O12345想退货',
        '这个茶杯怎么这么差劲',
        '查一下物流',
        '改一下地址',
        '开发票',
        '领优惠券',
        '这个手机壳质量太差了',
        '我的快递到哪了',
        '新地址是北京市朝阳区',
        '订单O98765开发票',
        '手机13800138000领优惠券',
        '订单O11111的物流',
        '收货地址是上海市浦东新区',
        '蓝牙耳机要退货',
        '订单O22222改地址为广州市天河区',
        '商品充电宝开发票',
        '收到的衣服有问题想退货',
        '路由器坏了申请退货',
        '订单O55555的护肤品物流',
        '我想退货',
        '帮我查一下订单O12345到哪了',
        '收货地址写错了改一下',
        '订单O66666需要开发票',
        '手机号13912345678领优惠券',
        '转人工客服',
    ]
    
    for text in test_cases:
        result = predictor.predict(text)
        print(f'Text: {text}')
        print(f'Intent: {result['intent']} (confidence: {result['confidence']:.2f})')
        print(f'Slots: {result['slots']}')
        print('---')