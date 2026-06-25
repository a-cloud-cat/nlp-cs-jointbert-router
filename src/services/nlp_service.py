from __future__ import annotations

import os
import sys
import random
import re
import json
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ChannelGroup, Prediction


CONFIDENCE_HIGH = 0.85
CONFIDENCE_MEDIUM = 0.4
MAX_CLARIFICATION_ROUNDS = 5
WELCOME_MESSAGE = "亲，您有什么需求？"

LEARNED_PRODUCTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'learned_products.json')

QUANTIFIERS = ['个', '件', '套', '只', '双', '条', '台', '瓶', '盒', '袋', '款', '款', '类', '种', '样', '些', '多', '各', '每']
ADJECTIVES = ['好', '坏', '新', '旧', '大', '小', '高', '矮', '长', '短', '胖', '瘦', '宽', '窄', '厚', '薄',
              '红', '绿', '蓝', '白', '黑', '黄', '紫', '粉', '灰', '棕',
              '漂亮', '好看', '难看', '可爱', '实用', '便宜', '贵', '划算',
              '质量', '问题', '破损', '损坏', '坏了', '有问题', '太差', '差劲',
              '满意', '不满意', '喜欢', '不喜欢', '想要', '需要', '想买', '收到']

COMMON_PRODUCTS = [
        '手机壳', '蓝牙耳机', '笔记本电脑', '充电宝', '数据线', '鼠标', '键盘', '显示器',
        '茶杯', '保温杯', '水杯', '水壶', '水瓶', '衣服', '鞋子', '包包', '化妆品', '护肤品', '手表', '耳机',
        '台灯', '书桌', '椅子', '柜子', '书架', '床', '沙发', '茶几', '桌子', '电脑', '手机', '平板', '音箱', '路由器', '硬盘',
        '充电器', '电池', 'U盘', '摄像头', '耳机套', 'T恤', '裤子', '帽子', '围巾',
        '手套', '袜子', '眼镜', '雨伞', '背包', '钱包', '皮带', '领带', '皮鞋',
        '运动鞋', '拖鞋', '睡衣', '内衣', '外套', '毛衣', '衬衫', '裙子', '短裤',
        '冰箱', '洗衣机', '空调', '电视', '微波炉', '烤箱', '电饭煲', '电水壶', '电风扇', '吹风机',
        '牙刷', '牙膏', '毛巾', '浴巾', '洗发水', '沐浴露', '洗面奶', '面霜', '面膜',
        '书籍', '课本', '教材', '文具', '笔', '本子', '书包', '文件夹', '计算器', '台灯', '书本',
        '锅', '碗', '盘子', '筷子', '勺子', '刀', '砧板', '保鲜盒',
        '窗帘', '地毯', '枕头', '被子', '床单', '被套', '蚊帐', '衣架',
        '自行车', '电动车', '汽车用品', '行李箱', '旅行包', '雨伞', '太阳镜',
        '零食', '食品', '水果', '蔬菜', '饮料', '牛奶', '面包', '饼干', '巧克力', '糖果',
        '坚果', '薯片', '方便面', '火腿肠', '罐头', '茶叶', '咖啡', '酸奶', '蛋糕', '冰淇淋',
        '娃娃', '玩具', '玩偶', '公仔', '毛绒玩具', '积木', '拼图', '模型', '手办'
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
                from model.inference import Predictor
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

    PRODUCT_KEYWORDS = ['柜', '桌', '椅', '床', '沙发', '茶几', '书架', '衣柜', '鞋柜', '电视柜', '书柜', '床头柜', '储物柜', '餐柜', '办公桌', '电脑桌',
                     '杯', '壶', '瓶', '碗', '盘', '锅', '勺', '筷',
                     '衣服', '鞋', '包', '帽', '裤', '裙', '衫', '袜', '手套',
                     '手机', '电脑', '平板', '耳机', '音箱', '键盘', '鼠标', '显示器',
                     '冰箱', '洗衣机', '空调', '电视', '微波炉', '烤箱', '电饭煲',
                     '书', '笔', '本', '文具',
                     '牙刷', '牙膏', '毛巾', '洗发水', '沐浴露', '化妆品',
                     '眼镜', '手表', '皮带', '领带',
                     '伞', '风扇', '台灯', '充电器', '电池']

    def extract_product(self, text: str) -> Optional[str]:
        exclude_words = ['退货', '物流', '发票', '优惠券', '地址', '订单', '手机', '太差', '差劲', '问题', '质量', '商品', '退款', '换货', '补发', '破损', '损坏', '有问题', '坏', '想退', '想换', '不行', '不好用', '不能用', '用不了', '没法用', '效果不好', '没用', '不太行', '不满意', '差劲', '糟糕']

        all_products = self.get_all_products() + sorted(self.PRODUCT_KEYWORDS, key=len, reverse=True)
        all_products = sorted(set(all_products), key=len, reverse=True)

        for product in all_products:
            pattern = r'(?<![\u4e00-\u9fa5])' + re.escape(product) + r'(?![\u4e00-\u9fa5])'
            if re.search(pattern, text):
                return product

        for product in all_products:
            if product in text:
                return product

        product_patterns = [
            r'(这个|那个|买的|收到的|商品)\s*([\u4e00-\u9fa5]{2,6})',
            r'(买|收到|购买|下单)\s*([\u4e00-\u9fa5]{2,6})',
        ]

        for pattern in product_patterns:
            match = re.search(pattern, text)
            if match:
                product = match.group(2) if len(match.groups()) > 1 else match.group(1)
                if product not in exclude_words and len(product) >= 2:
                    for neg in ['不行', '不好用', '不能用', '用不了', '没法用', '效果不好', '没用', '不太行', '不满意', '差劲', '糟糕']:
                        if product.endswith(neg):
                            product = product[:-len(neg)]
                    if len(product) >= 2:
                        return product

        suffix_patterns = [
            r'(这个|那个)\s*([\u4e00-\u9fa5]{1,4})(柜|桌|椅|床|杯|壶|瓶|碗|盘|锅)',
            r'([\u4e00-\u9fa5]{1,4})(柜|桌|椅|床|杯|壶|瓶|碗|盘|锅)\s*(坏|破损|损坏|有问题|想退|想换|质量)',
        ]

        for pattern in suffix_patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) >= 3:
                    candidate = match.group(2) + match.group(3)
                    if candidate not in exclude_words and len(candidate) >= 2:
                        return candidate

        product_suffix_patterns = [
            r'(?<![\u4e00-\u9fa5])([\u4e00-\u9fa5]{1,3})(柜|桌|椅|床|杯|壶|瓶|碗|盘|锅)(?![\u4e00-\u9fa5])',
            r'(?<![\u4e00-\u9fa5])([\u4e00-\u9fa5]{1,2})(鞋|包|帽|裤|裙|衫|袜|书|笔|本|伞|镜|表|灯)(?![\u4e00-\u9fa5])',
        ]

        for pattern in product_suffix_patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(0)
                if candidate not in exclude_words and len(candidate) >= 2:
                    return candidate

        return None

    def _clean_product_name(self, text: str) -> Optional[str]:
        text = text.strip()

        exclude_patterns = [
            r'(退货|退款|换货|补发|开发票|物流|快递|质量|问题|破损|损坏|坏|想退|想换|有问题)',
            r'(买|收到|购买|下单|这个|那个)',
        ]
        for pattern in exclude_patterns:
            text = re.sub(pattern, '', text)

        text = re.sub(r'\s+', '', text)

        text = re.sub(r'^[\u4e00-\u4e9f]{1,2}(个|件|套|只|双|条|台|瓶|盒|袋|款|类|种|样|些|多|各|每)', '', text)
        text = re.sub(r'(个|件|套|只|双|条|台|瓶|盒|袋|款|类|种|样|些|多|各|每)([\u4e00-\u9fa5])', r'\2', text)

        text = re.sub(r'(好|坏|新|旧|大|小|高|矮|长|短|胖|瘦|宽|窄|厚|薄)(的)', '', text)
        text = re.sub(r'(红|绿|蓝|白|黑|黄|紫|粉|灰|棕)(的)', '', text)
        text = re.sub(r'(漂亮|好看|难看|可爱|实用|便宜|贵|划算)(的)', '', text)

        text = re.sub(r'^[的是了我你他她它那这想]+', '', text)
        text = re.sub(r'[的是了我你他她它]+$', '', text)

        text = text.strip()

        if len(text) >= 2:
            return text
        return None

    def _strip_adjectives(self, text: str) -> Optional[str]:
        adj_patterns = [
            r'^(好|坏|新|旧|大|小|高|矮|长|短|胖|瘦|宽|窄|厚|薄)',
            r'(好|坏|新|旧|大|小|高|矮|长|短|胖|瘦|宽|窄|厚|薄)$',
            r'^(红|绿|蓝|白|黑|黄|紫|粉|灰|棕)',
            r'(红|绿|蓝|白|黑|黄|紫|粉|灰|棕)$',
            r'^(漂亮|好看|难看|可爱|实用|便宜|贵|划算)$',
            r'^[的是了我你他她它]+',
            r'[的是了我你他她它]+$',
            r'^那{1,2}',
            r'^(这|那)$',
        ]

        while True:
            original = text
            for pattern in adj_patterns:
                text = re.sub(pattern, '', text)
            if text == original:
                break

        text = text.strip()
        return text if len(text) >= 2 else None

    def learn_product(self, user_input: str) -> Optional[str]:
        cleaned = self._clean_product_name(user_input)
        if not cleaned:
            return None

        if cleaned in COMMON_PRODUCTS:
            return cleaned

        learned = self._load_learned_products()
        if cleaned not in learned:
            learned.append(cleaned)
            self._save_learned_products(learned)

        return cleaned

    def _load_learned_products(self) -> list:
        if not os.path.exists(LEARNED_PRODUCTS_FILE):
            return []
        try:
            with open(LEARNED_PRODUCTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def _save_learned_products(self, products: list) -> None:
        os.makedirs(os.path.dirname(LEARNED_PRODUCTS_FILE), exist_ok=True)
        with open(LEARNED_PRODUCTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)

    def get_all_products(self) -> list:
        learned = self._load_learned_products()
        return sorted(set(COMMON_PRODUCTS + learned), key=len, reverse=True)

    INTENT_PRIORITY = {
        '质量问题': 10,
        '商品投诉': 9,
        '商家投诉': 8,
        '退货申请': 7,
        '退款申请': 6,
        '换货申请': 5,
        '货不对板': 4,
        '少发漏发': 3,
        '拒收': 2,
        '售后维权': 1,
        '丢件破损': -1,
        '物流查询': -2,
        '查快递': -3,
        '发货时效': -4,
    }

    def detect_intent_by_rules(self, text: str) -> Tuple[Optional[str], float]:
        from registry import get_registry

        registry = get_registry()
        matched_channels = registry.find_channels_by_keyword(text)

        if matched_channels:
            best_channel = None
            best_score = -1
            best_confidence = 0.0

            for channel in matched_channels:
                matched_count = sum(1 for kw in channel.keywords if kw in text)
                confidence = matched_count / len(channel.keywords)
                priority = self.INTENT_PRIORITY.get(channel.intent, 0)
                score = priority + confidence

                if score > best_score:
                    best_score = score
                    best_confidence = confidence
                    best_channel = channel

            if best_channel:
                return best_channel.intent, best_confidence

        return None, 0.0

    def predict(self, text: str, session_id: Optional[str] = None) -> Prediction:
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

        context_text = text
        if session_id:
            from services.session_service import get_session_service
            session_service = get_session_service()
            session_history = session_service.get_conversation_summary(session_id)
            if session_history:
                context_text = f"{session_history}\n用户最新输入: {text}"

        rule_intent, rule_confidence = self.detect_intent_by_rules(text)
        context_intent, context_confidence = self.detect_intent_by_rules(context_text)

        if context_intent and context_confidence > rule_confidence:
            rule_intent = context_intent
            rule_confidence = context_confidence

        model_intent = None
        model_confidence = 0.0
        model_slots = {}

        if self._predictor:
            try:
                result = self._predictor.predict(text)
                model_intent = result['intent']
                model_confidence = result['confidence']
                model_slots = result.get('slots', {})
            except Exception:
                model_intent = None

        # ======================核心修改：模型结果优先，规则仅兜底======================
        final_intent = None
        confidence = 0.0
        # 模型可用且置信度达标，优先使用模型结果
        if model_intent is not None and model_confidence > 0.7:
            final_intent = model_intent
            confidence = model_confidence
        else:
            # 模型失效/分数低，降级使用关键词规则
            if rule_intent:
                final_intent = rule_intent
                confidence = rule_confidence
            else:
                final_intent = 'default'
                confidence = 0.0
        # ==========================================================================

        if session_id:
            session_service = get_session_service()
            prev_slots = session_service.get_slots(session_id)
            if prev_slots and '商品名' in prev_slots:
                if final_intent in ['退货申请', '退款申请', '换货申请', '质量问题', '货不对板', '商品投诉']:
                    if confidence < CONFIDENCE_HIGH:
                        confidence = min(CONFIDENCE_HIGH, confidence + 0.2)

        rule_slots = self.extract_slots(text)
        product_name = None

        if final_intent in ['退货申请', '退款申请', '换货申请', '质量问题', '货不对板', '开发票', '物流查询', '查快递', '商品投诉']:
            product_name = self.extract_product(text)

            if not product_name and session_id:
                session_service = get_session_service()
                prev_slots = session_service.get_slots(session_id)
                if prev_slots and '商品名' in prev_slots:
                    product_name = prev_slots['商品名']

        final_slots = {}
        for slot_name in ['订单号', '手机号', '商品名', '收货地址', '发票抬头']:
            if slot_name in model_slots:
                final_slots[slot_name] = model_slots[slot_name]
            elif slot_name == '商品名' and product_name:
                final_slots['商品名'] = product_name
            elif slot_name in rule_slots:
                final_slots[slot_name] = rule_slots[slot_name]
            elif session_id:
                session_service = get_session_service()
                prev_slots = session_service.get_slots(session_id)
                if slot_name in prev_slots:
                    final_slots[slot_name] = prev_slots[slot_name]

        group = self._get_group_by_intent(final_intent)

        if session_id:
            from services.session_service import get_session_service
            session_service = get_session_service()
            session_service.update_slots(session_id, final_slots)
            session_service.append_intent(session_id, final_intent, confidence)

        return Prediction(
            intent=final_intent,
            slots=final_slots,
            confidence=confidence,
            group=group,
            clarification_needed=confidence < CONFIDENCE_HIGH,
            clarification_question=self._get_clarification_question(final_intent)
        )

    def _get_group_by_intent(self, intent: str) -> ChannelGroup:
        from registry import get_registry

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