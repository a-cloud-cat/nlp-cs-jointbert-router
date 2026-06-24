from __future__ import annotations

import random
import re
from typing import Dict, Optional, Tuple

from ..models import ChannelGroup, Prediction


CONFIDENCE_HIGH = 0.85
CONFIDENCE_MEDIUM = 0.4
MAX_CLARIFICATION_ROUNDS = 5
WELCOME_MESSAGE = "亲，您有什么需求？"

COMMON_PRODUCTS = [
    '手机壳', '蓝牙耳机', '笔记本电脑', '充电宝', '数据线', '鼠标', '键盘', '显示器',
    '茶杯', '保温杯', '衣服', '鞋子', '包包', '化妆品', '护肤品', '手表', '耳机',
    '台灯', '书桌', '椅子', '电脑', '手机', '平板', '音箱', '路由器', '硬盘',
    '充电器', '电池', 'U盘', '摄像头', '耳机套', 'T恤', '裤子', '帽子', '围巾',
    '手套', '袜子', '眼镜', '雨伞', '背包', '钱包', '皮带', '领带', '皮鞋',
    '运动鞋', '拖鞋', '睡衣', '内衣', '外套', '毛衣', '衬衫', '裙子', '短裤'
]

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


class NLPService:
    def __init__(self):
        self._predictor = None

    def _init_predictor(self):
        if self._predictor is None:
            try:
                from predict import Predictor
                self._predictor = Predictor()
            except Exception:
                self._predictor = None

    def extract_slots(self, text: str) -> Dict[str, str]:
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

    def extract_product(self, text: str) -> Optional[str]:
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

    def detect_intent_by_rules(self, text: str) -> Tuple[Optional[str], float]:
        from ..registry import get_registry

        registry = get_registry()
        matched_channels = registry.find_channels_by_keyword(text)

        if matched_channels:
            channel = matched_channels[0]
            matched_count = sum(1 for kw in channel.keywords if kw in text)
            confidence = matched_count / len(channel.keywords)
            return channel.intent, confidence

        return None, 0.0

    def predict(self, text: str) -> Prediction:
        text = text.strip()
        if not text:
            return Prediction(
                intent='default',
                slots={},
                confidence=0.0,
                group=ChannelGroup.FALLBACK,
                clarification_needed=True,
                clarification_question=self._get_clarification_question('default')
            )

        self._init_predictor()

        rule_intent, rule_confidence = self.detect_intent_by_rules(text)
        model_intent = None
        model_confidence = 0.0

        if self._predictor:
            try:
                result = self._predictor.predict(text)
                model_intent = result['intent']
                model_confidence = result['confidence']
            except Exception:
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

        rule_slots = self.extract_slots(text)
        product_name = None

        if final_intent in ['退货申请', '退款申请', '换货申请', '质量问题', '货不对板', '开发票', '物流查询', '查快递']:
            product_name = self.extract_product(text)

        final_slots = {}
        for slot_name in ['订单号', '手机号', '商品名', '收货地址', '发票抬头']:
            if slot_name == '商品名' and product_name:
                final_slots['商品名'] = product_name
            elif slot_name in rule_slots:
                final_slots[slot_name] = rule_slots[slot_name]

        group = self._get_group_by_intent(final_intent)

        return Prediction(
            intent=final_intent,
            slots=final_slots,
            confidence=confidence,
            group=group,
            clarification_needed=confidence < CONFIDENCE_HIGH,
            clarification_question=self._get_clarification_question(final_intent)
        )

    def _get_group_by_intent(self, intent: str) -> ChannelGroup:
        from ..registry import get_registry

        registry = get_registry()
        channel = registry.get_channel_by_intent(intent)
        if channel:
            return channel.group
        return ChannelGroup.FALLBACK

    def _get_clarification_question(self, intent: str) -> str:
        questions = CLARIFICATION_QUESTIONS.get(intent, CLARIFICATION_QUESTIONS['default'])
        if isinstance(questions, list):
            return random.choice(questions)
        return questions

    def classify_confidence(self, confidence: float) -> str:
        if confidence >= CONFIDENCE_HIGH:
            return 'high'
        elif confidence >= CONFIDENCE_MEDIUM:
            return 'medium'
        else:
            return 'low'


_nlp_service: Optional[NLPService] = None


def get_nlp_service() -> NLPService:
    global _nlp_service
    if _nlp_service is None:
        _nlp_service = NLPService()
    return _nlp_service