from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from models import ChannelGroup, ServiceChannel


@dataclass(frozen=True)
class IntentRoute:
    intent: str
    channel: ServiceChannel


class ServiceRegistry:
    def __init__(self):
        self._channels: Dict[str, ServiceChannel] = {}
        self._intent_map: Dict[str, ServiceChannel] = {}
        self._keyword_map: Dict[str, List[str]] = {}

    def register(self, channel: ServiceChannel) -> None:
        self._channels[channel.name] = channel
        self._intent_map[channel.intent] = channel
        for keyword in channel.keywords:
            if keyword not in self._keyword_map:
                self._keyword_map[keyword] = []
            self._keyword_map[keyword].append(channel.name)

    def get_channel(self, name: str) -> Optional[ServiceChannel]:
        return self._channels.get(name)

    def get_channel_by_intent(self, intent: str) -> Optional[ServiceChannel]:
        return self._intent_map.get(intent)

    def find_channels_by_keyword(self, text: str) -> List[ServiceChannel]:
        matched_names = set()
        for keyword, names in self._keyword_map.items():
            if keyword in text:
                matched_names.update(names)
        return [self._channels[name] for name in matched_names if name in self._channels]

    def get_channels_by_group(self, group: ChannelGroup) -> List[ServiceChannel]:
        return [channel for channel in self._channels.values() if channel.group == group]

    def list_all_channels(self) -> List[ServiceChannel]:
        return list(self._channels.values())

    def list_all_intents(self) -> List[str]:
        return list(self._intent_map.keys())


_default_registry: Optional[ServiceRegistry] = None


def get_registry() -> ServiceRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = _build_default_registry()
    return _default_registry


def _build_default_registry() -> ServiceRegistry:
    registry = ServiceRegistry()

    transaction_keywords = {
        '退货申请': ('退货', '想退', '要退', '退货退款', '申请退货', '退回去', '想把货退了', '办理退货'),
        '退款申请': ('退款', '退钱', '仅退款', '退货款', '只想退款', '不退货退款', '只退钱'),
        '换货申请': ('换货', '换一个', '更换', '换一件', '换尺码', '换颜色', '申请换货'),
        '拒收': ('拒收', '不要了', '拒签', '不想签收', '拒绝签收', '退回包裹'),
        '售后维权': ('维权', '投诉维权', '申请维权', '维权申请', '售后投诉'),
        '商品投诉': ('投诉', '投诉商品', '商品投诉', '商品有问题', '产品质量'),
        '商家投诉': ('商家态度', '投诉商家', '商家恶劣', '服务态度', '客服态度'),
        '质量问题': ('质量差', '质量问题', '有问题', '坏了', '质量不好', '做工差', '有缺陷', '发霉', '发霉了', '发霉的', '变质', '过期', '破损', '损坏', '开裂', '断裂', '变形', '漏水', '漏油', '异味', '异味重', '色差', '尺寸不符', '材质差', '甲醛', '甲醛味', '气味重', '刺鼻', '有毒', '有害物质', '过敏', '掉色', '褪色', '脱皮', '开胶', '生锈', '划痕', '污渍'),
        '货不对板': ('货不对板', '发错货', '不是这个', '与描述不符', '描述不一致'),
        '少发漏发': ('少发', '漏发', '没发全', '少了一件', '漏寄了', '少寄了'),
    }

    logistics_keywords = {
        '物流查询': ('物流', '查物流', '物流进度', '物流信息', '快递进度'),
        '查快递': ('快递', '查快递', '快递单号', '快递到哪', '我的快递'),
        '发货时效': ('发货', '什么时候发货', '发货时间', '多久发货', '还没发货'),
        '丢件破损': ('丢件', '破损', '包裹破了', '快递丢了', '包裹丢了'),
        '改地址': ('改地址', '地址错了', '换地址', '变更地址', '修改地址'),
        '改手机号': ('改手机号', '手机号错了', '换号码', '修改手机号'),
        '改配送': ('改配送', '配送地址', '驿站', '改驿站', '换配送站'),
    }

    invoice_keywords = {
        '开发票': ('开发票', '发票', '开票', '申请开票', '需要发票'),
        '改发票抬头': ('发票抬头', '改抬头', '抬头错了', '修改抬头'),
        '补开发票': ('补开发票', '补发票', '发票没开', '忘记开发票'),
        '领优惠券': ('优惠券', '领券', '优惠券领取', '领取优惠券'),
        '店铺券': ('店铺券', '店铺优惠券', '店铺优惠'),
        '价格补差': ('补差', '补差价', '价格差', '差价'),
    }

    fallback_keywords = {
        '转人工': ('转人工', '人工客服', '找人工', '人工服务', '人工帮忙'),
        'default': (),
    }

    for name, keywords in transaction_keywords.items():
        registry.register(ServiceChannel(
            name=name,
            intent=name,
            group=ChannelGroup.TRANSACTION_AFTER_SALES,
            api=f'/api/services/{name.replace(" ", "_").lower()}',
            method='POST',
            description=_get_channel_description(name),
            keywords=keywords,
            required_slots=('订单号', '商品名'),
        ))

    for name, keywords in logistics_keywords.items():
        registry.register(ServiceChannel(
            name=name,
            intent=name,
            group=ChannelGroup.LOGISTICS_DELIVERY,
            api=f'/api/services/{name.replace(" ", "_").lower()}',
            method='GET' if name in ('物流查询', '查快递', '发货时效') else 'POST',
            description=_get_channel_description(name),
            keywords=keywords,
            required_slots=('订单号',),
        ))

    for name, keywords in invoice_keywords.items():
        registry.register(ServiceChannel(
            name=name,
            intent=name,
            group=ChannelGroup.INVOICE_DISCOUNT,
            api=f'/api/services/{name.replace(" ", "_").lower()}',
            method='POST',
            description=_get_channel_description(name),
            keywords=keywords,
            required_slots=('订单号',),
        ))

    for name, keywords in fallback_keywords.items():
        registry.register(ServiceChannel(
            name=name,
            intent=name,
            group=ChannelGroup.FALLBACK,
            api=f'/api/services/{name.replace(" ", "_").lower()}',
            method='POST',
            description=_get_channel_description(name),
            keywords=keywords,
        ))

    return registry


def _get_channel_description(name: str) -> str:
    descriptions = {
        '退货申请': '处理商品退货申请',
        '退款申请': '处理仅退款申请',
        '换货申请': '处理商品换货申请',
        '拒收': '处理拒收申请',
        '售后维权': '处理售后维权申请',
        '商品投诉': '处理商品投诉',
        '商家投诉': '处理商家投诉',
        '质量问题': '处理质量问题反馈',
        '货不对板': '处理货不对板问题',
        '少发漏发': '处理少发漏发问题',
        '物流查询': '查询物流进度',
        '查快递': '查询快递信息',
        '发货时效': '查询发货时效',
        '丢件破损': '处理丢件破损',
        '改地址': '修改收货地址',
        '改手机号': '修改联系电话',
        '改配送': '修改配送方式',
        '开发票': '开具电子发票',
        '改发票抬头': '修改发票抬头',
        '补开发票': '补开发票',
        '领优惠券': '领取优惠券',
        '店铺券': '领取店铺券',
        '价格补差': '处理价格补差',
        '转人工': '转接人工客服',
        'default': '默认处理通道',
    }
    return descriptions.get(name, '未知服务')