import torch
import os
import re
import random
from typing import Dict, Optional

INTENT_LABELS = [
    "退货申请", "退款申请", "换货申请", "拒收", "售后维权",
    "商品投诉", "商家投诉", "质量问题", "货不对板", "少发漏发",
    "物流查询", "查快递", "发货时效", "丢件破损",
    "改地址", "改手机号", "改配送",
    "开发票", "改发票抬头", "补开发票",
    "领优惠券", "店铺券", "价格补差",
    "转人工", "default"
]

SLOT_IDX_MAP = {
    0: 'O',
    1: 'B-订单号',
    2: 'I-订单号',
    3: 'B-手机号',
    4: 'I-手机号',
    5: 'B-商品名',
    6: 'I-商品名',
    7: 'B-收货地址',
    8: 'I-收货地址',
    9: 'B-发票抬头',
    10: 'I-发票抬头',
}

BERT_MODEL_NAME = "bert-base-chinese"
CHECKPOINT_DIR = "./checkpoints"
MAX_SEQ_LEN = 128

CONFIDENCE_HIGH = 0.85
CONFIDENCE_MEDIUM = 0.4
MAX_CLARIFICATION_ROUNDS = 5

COMMON_PRODUCTS = [
    '手机壳', '蓝牙耳机', '笔记本电脑', '充电宝', '数据线', '鼠标', '键盘', '显示器',
    '茶杯', '保温杯', '衣服', '鞋子', '包包', '化妆品', '护肤品', '手表', '耳机',
    '台灯', '书桌', '椅子', '电脑', '手机', '平板', '音箱', '路由器', '硬盘',
    '充电器', '电池', 'U盘', '摄像头', '耳机套', 'T恤', '裤子', '帽子', '围巾',
    '手套', '袜子', '眼镜', '雨伞', '背包', '钱包', '皮带', '领带', '皮鞋',
    '运动鞋', '拖鞋', '睡衣', '内衣', '外套', '毛衣', '衬衫', '裙子', '短裤'
]

INTENT_KEYWORDS = {
    '退货申请': ['退货', '想退', '要退', '退货退款', '申请退货', '退回去', '想把货退了', '申请退款退货', '办理退货', '退货申请', '退货款'],
    '退款申请': ['退款', '退钱', '仅退款', '退货款', '只想退款', '不退货退款', '只退钱', '申请退款', '退款申请'],
    '换货申请': ['换货', '换一个', '更换', '换一件', '换尺码', '换颜色', '申请换货', '想换个', '换型号'],
    '拒收': ['拒收', '不要了', '拒签', '不想签收', '拒绝签收', '退回包裹'],
    '售后维权': ['维权', '投诉维权', '申请维权', '维权申请', '售后投诉', '申请售后'],
    '商品投诉': ['投诉', '投诉商品', '商品投诉', '商品有问题', '产品质量', '商品质量', '有瑕疵'],
    '商家投诉': ['商家态度', '投诉商家', '商家恶劣', '服务态度', '客服态度', '卖家态度'],
    '质量问题': ['质量差', '质量问题', '有问题', '坏了', '差劲', '瑕疵', '质量不好', '质量太差', '做工差', '有缺陷', '损坏'],
    '货不对板': ['货不对板', '发错货', '不是这个', '与描述不符', '描述不一致', '收到的不是', '发错了', '寄错了'],
    '少发漏发': ['少发', '漏发', '没发全', '少了一件', '漏寄了', '少寄了', '缺少', '没收到'],
    '物流查询': ['物流', '查物流', '物流进度', '物流信息', '快递进度', '查快递信息', '物流到哪了'],
    '查快递': ['快递', '查快递', '快递单号', '快递到哪', '我的快递', '快递到了吗', '快递状态'],
    '发货时效': ['发货', '什么时候发货', '发货时间', '多久发货', '啥时候发货', '发货吗', '还没发货'],
    '丢件破损': ['丢件', '破损', '包裹破了', '快递丢了', '包裹丢了', '快递破损', '箱子破了', '物品损坏'],
    '改地址': ['改地址', '地址错了', '换地址', '变更地址', '修改地址', '地址写错了', '改收货地址'],
    '改手机号': ['改手机号', '手机号错了', '换号码', '修改手机号', '手机号写错', '改电话'],
    '改配送': ['改配送', '配送地址', '驿站', '改驿站', '换配送站', '改自提点'],
    '开发票': ['开发票', '发票', '开票', '申请开票', '需要发票', '要开发票', '开个发票'],
    '改发票抬头': ['发票抬头', '改抬头', '抬头错了', '修改抬头', '发票抬头错误', '更正抬头'],
    '补开发票': ['补开发票', '补发票', '发票没开', '忘记开发票', '补开'],
    '领优惠券': ['优惠券', '领券', '优惠券领取', '领取优惠券', '有优惠券吗', '券在哪', '领优惠'],
    '店铺券': ['店铺券', '店铺优惠券', '店铺优惠', '店内券'],
    '价格补差': ['补差', '补差价', '价格差', '差价', '退差价', '价格少退'],
    '转人工': ['转人工', '人工客服', '找人工', '人工服务', '人工帮忙', '联系人工', '要人工']
}

INTENT_GROUPS = {
    '退货申请': '交易售后类',
    '退款申请': '交易售后类',
    '换货申请': '交易售后类',
    '拒收': '交易售后类',
    '售后维权': '交易售后类',
    '商品投诉': '交易售后类',
    '商家投诉': '交易售后类',
    '质量问题': '交易售后类',
    '货不对板': '交易售后类',
    '少发漏发': '交易售后类',
    '物流查询': '物流配送类',
    '查快递': '物流配送类',
    '发货时效': '物流配送类',
    '丢件破损': '物流配送类',
    '改地址': '物流配送类',
    '改手机号': '物流配送类',
    '改配送': '物流配送类',
    '开发票': '票据优惠类',
    '改发票抬头': '票据优惠类',
    '补开发票': '票据优惠类',
    '领优惠券': '票据优惠类',
    '店铺券': '票据优惠类',
    '价格补差': '票据优惠类',
    '转人工': '兜底通道',
    'default': '兜底通道'
}

CLARIFICATION_QUESTIONS = {
    '退货申请': ['请问您是想退货退款，还是仅退款呢？', '请问您收到的商品是有质量问题，还是不喜欢想退货？', '请问您是想申请退货退款，还是只退款不退货呢？'],
    '退款申请': ['请问您是想仅退款，还是退货退款呢？', '请问您是因为商品问题退款，还是其他原因？', '请问您希望退全款还是部分金额？'],
    '换货申请': ['请问您是想换个尺码、颜色，还是换个型号？', '请问您是收到了错误商品想换，还是想换其他款式？', '请问您希望更换同款商品，还是换其他商品呢？'],
    '拒收': ['请问您是因为商品问题拒收，还是单纯不想要了？', '请问您已经收到包裹了吗？还是想直接拒签？'],
    '售后维权': ['请问您是想投诉商家，还是申请售后维权呢？', '请问您遇到了什么问题需要维权？'],
    '商品投诉': ['请问您是想投诉商品质量，还是货不对板的问题？', '请问您是收到了损坏的商品，还是商品与描述不符？'],
    '商家投诉': ['请问您是对商家的服务态度不满，还是对处理结果不满意？', '请问商家是哪里让您不满意了呢？'],
    '质量问题': ['请问商品是哪里出现了质量问题？', '请问您收到的商品是有破损，还是做工问题？', '请问商品是刚收到就坏了，还是使用后出现问题？'],
    '货不对板': ['请问是收到的商品与图片不符，还是颜色尺码不对？', '请问商家发错了商品，还是商品本身与描述不一致？'],
    '少发漏发': ['请问是少发了商品，还是漏发了配件？', '请问您收到的包裹里缺少了什么？'],
    '物流查询': ['请问您是想查询物流进度，还是需要修改配送地址？', '请问您知道快递单号吗？需要帮您查一下物流信息吗？', '请问您是想查订单物流，还是咨询发货时间？'],
    '查快递': ['请问您是想查快递到哪了，还是有其他物流问题？', '请问您需要查询哪个订单的快递？'],
    '发货时效': ['请问您是想问什么时候发货，还是有发货延迟的问题？', '请问您下单后多久了还没发货呢？'],
    '丢件破损': ['请问是包裹丢了，还是收到的包裹有破损？', '请问是外包装破损，还是里面的商品损坏了？'],
    '改地址': ['请问您是想修改收货地址，还是修改手机号码？', '请问您是下单后地址写错了，还是想更换配送地址？'],
    '改手机号': ['请问您是想修改联系电话，还是收货地址呢？', '请问您的手机号是写错了，还是需要更换？'],
    '改配送': ['请问您是想改配送地址，还是改配送驿站？', '请问您是想换个自提点，还是修改配送方式？'],
    '开发票': ['请问您是想开具电子发票，还是纸质发票？', '请问您需要开具个人发票还是公司发票？', '请问您需要开具电子发票还是增值税发票？'],
    '改发票抬头': ['请问您是想修改发票抬头，还是重新开具发票？', '请问发票抬头需要改成什么名称？'],
    '补开发票': ['请问您是想补开发票，还是修改发票信息？', '请问您需要补开哪个订单的发票？'],
    '领优惠券': ['请问您是想领取店铺优惠券，还是活动优惠券？', '请问您是想领满减券，还是无门槛优惠券呢？'],
    '店铺券': ['请问您是想领取店铺券，还是参与店铺活动？', '请问您想领取哪类店铺优惠券？'],
    '价格补差': ['请问您是想申请价格补差，还是其他问题？', '请问您购买的商品降价了需要补差吗？'],
    '转人工': ['请问您确定需要转接人工客服吗？', '请问您遇到了什么问题需要人工帮助？'],
    'default': ['请问您主要是遇到了什么问题？是退货、物流、发票还是其他问题呢？', '请问我可以帮您处理什么问题？比如退货、查物流、开发票等', '请问您是想咨询售后问题，还是其他服务呢？']
}


class Predictor:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_model()

    def _init_model(self):
        try:
            from transformers import BertTokenizer
            from model import JointBERT

            self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
            self.model = JointBERT()

            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                print('Model loaded successfully')
            else:
                print(f'Warning: Model checkpoint not found at {checkpoint_path}')
                self.model = None
        except Exception as e:
            print(f'Failed to load model: {e}')
            self.tokenizer = None
            self.model = None

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

        address_keywords = ['北京市', '上海市', '广州市', '深圳市', '杭州市', '南京市', '成都市', '武汉市', '重庆市', '天津市', '苏州市', '西安市', '长沙市', '郑州市', '东莞市', '佛山市', '厦门市', '沈阳市', '青岛市', '宁波市', '合肥市', '无锡市', '昆明市', '大连市', '济南市', '南宁市', '哈尔滨市', '福州市', '长春市', '石家庄市']
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

        invoice_pattern = r'(发票抬头|抬头)\s*[：:]\s*([\u4e00-\u9fa5]{2,20})'
        invoice_match = re.search(invoice_pattern, text)
        if invoice_match:
            slots['发票抬头'] = invoice_match.group(2)

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
        intent_scores = {}

        for intent, keywords in INTENT_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += 1
            if score > 0:
                intent_scores[intent] = score

        if intent_scores:
            return max(intent_scores, key=intent_scores.get), max(intent_scores.values()) / len(INTENT_KEYWORDS[max(intent_scores, key=intent_scores.get)])
        return None, 0.0

    def predict(self, text):
        text = text.strip()
        if not text:
            return {'intent': 'default', 'slots': {}, 'confidence': 0.0, 'group': '兜底通道'}

        rule_intent, rule_confidence = self.detect_intent_by_rules(text)
        model_intent = None
        model_confidence = 0.0

        if self.model and self.tokenizer:
            try:
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
                    intent_logits, _ = self.model(input_ids, attention_mask)

                intent_probs = torch.softmax(intent_logits, dim=1)
                max_prob, intent_idx = torch.max(intent_probs, dim=1)
                model_intent = INTENT_LABELS[intent_idx]
                model_confidence = max_prob.item()
            except Exception as e:
                print(f'Model prediction error: {e}')
                model_intent = None

        if rule_intent:
            if model_intent and model_confidence > 0.7:
                if model_intent == rule_intent:
                    final_intent = model_intent
                    confidence = max(model_confidence, rule_confidence)
                else:
                    final_intent = rule_intent
                    confidence = rule_confidence
            else:
                final_intent = rule_intent
                confidence = rule_confidence
        elif model_intent:
            final_intent = model_intent
            confidence = model_confidence
        else:
            final_intent = 'default'
            confidence = 0.0

        rule_slots = self.extract_slots_by_rules(text)
        product_name = None

        if final_intent in ['退货申请', '退款申请', '换货申请', '质量问题', '货不对板', '开发票', '物流查询', '查快递']:
            product_name = self.extract_product(text)

        final_slots = {}
        for slot_name in ['订单号', '手机号', '商品名', '收货地址', '发票抬头']:
            if slot_name == '商品名' and product_name:
                final_slots['商品名'] = product_name
            elif slot_name in rule_slots:
                final_slots[slot_name] = rule_slots[slot_name]

        questions = CLARIFICATION_QUESTIONS.get(final_intent, CLARIFICATION_QUESTIONS['default'])
        if isinstance(questions, list):
            clarification_question = random.choice(questions)
        else:
            clarification_question = questions

        return {
            'intent': final_intent,
            'slots': final_slots,
            'confidence': confidence,
            'group': INTENT_GROUPS.get(final_intent, '兜底通道'),
            'clarification_needed': confidence < CONFIDENCE_HIGH,
            'clarification_question': clarification_question
        }

    def classify_confidence(self, confidence):
        if confidence >= CONFIDENCE_HIGH:
            return 'high'
        elif confidence >= CONFIDENCE_MEDIUM:
            return 'medium'
        else:
            return 'low'


if __name__ == '__main__':
    predictor = Predictor()

    test_cases = [
        '订单O12345想退货',
        '这个茶杯质量太差了',
        '查一下物流',
        '改一下地址',
        '开发票',
        '领优惠券',
        '我的快递到哪了',
        '新地址是北京市朝阳区',
        '订单O98765开发票',
        '手机13800138000领优惠券',
        '蓝牙耳机要退货',
        '订单O22222改地址为广州市天河区',
        '收到的衣服有问题想退货',
        '路由器坏了申请退货',
        '我想退货',
        '帮我查一下订单O12345到哪了',
        '收货地址写错了改一下',
        '订单O66666需要开发票',
        '手机号13912345678领优惠券',
        '转人工客服',
        '货不对板',
        '商家态度恶劣',
        '少发了一件',
        '包裹破损了',
        '发票抬头错了',
        '补开发票',
        '价格补差',
        '店铺优惠券',
    ]

    for text in test_cases:
        result = predictor.predict(text)
        print(f'Text: {text}')
        print(f'Intent: {result['intent']} (group: {result['group']}, confidence: {result['confidence']:.2f})')
        print(f'Slots: {result['slots']}')
        print(f'Clarification needed: {result['clarification_needed']}')
        print('---')
