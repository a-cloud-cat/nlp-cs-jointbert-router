from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Tuple, Any
from utils.errors import ServiceError, PermissionError
from utils.hooks import HookRunner, HookContext, HookResult


@dataclass(frozen=True)
class ServiceChannel:
    name: str
    intent: str
    description: str
    keywords: Tuple[str, ...]
    required_slots: Tuple[str, ...]
    optional_slots: Tuple[str, ...]
    handler: Optional[Callable[..., str]] = None
    permission_required: bool = False
    role_required: Optional[str] = None
    is_active: bool = True

    def requires_slot(self, slot_name: str) -> bool:
        return slot_name in self.required_slots


@dataclass(frozen=True)
class ToolExecution:
    channel: ServiceChannel
    slots: Dict[str, str]
    handled: bool
    message: str
    confidence: float = 0.0
    error_code: Optional[str] = None

    def is_success(self) -> bool:
        return self.handled and not self.error_code


@dataclass(frozen=True)
class PermissionContext:
    user_role: str = 'user'
    allowed_channels: Optional[List[str]] = None
    blocked_channels: Optional[List[str]] = None

    def allows(self, channel: ServiceChannel) -> bool:
        if self.blocked_channels and channel.name in self.blocked_channels:
            return False
        if self.allowed_channels and channel.name not in self.allowed_channels:
            return False
        if channel.role_required and self.user_role != channel.role_required:
            return False
        return True


@dataclass(frozen=True)
class UsageSummary:
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    successful_routes: int = 0

    def add_turn(self, input_text: str, output_text: str, success: bool = False) -> 'UsageSummary':
        return UsageSummary(
            input_tokens=self.input_tokens + len(input_text),
            output_tokens=self.output_tokens + len(output_text),
            requests=self.requests + 1,
            successful_routes=self.successful_routes + (1 if success else 0)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'requests': self.requests,
            'successful_routes': self.successful_routes,
            'success_rate': self.successful_routes / self.requests if self.requests > 0 else 0
        }


class UsageTracker:
    def __init__(self):
        self._summary: UsageSummary = UsageSummary()
        self._intent_stats: Dict[str, int] = {}

    def record(self, input_text: str, output_text: str, success: bool = False, intent: str = "") -> None:
        self._summary = self._summary.add_turn(input_text, output_text, success)
        if intent:
            self._intent_stats[intent] = self._intent_stats.get(intent, 0) + 1

    def get_summary(self) -> UsageSummary:
        return self._summary

    def get_intent_stats(self) -> Dict[str, int]:
        return self._intent_stats

    def reset(self) -> None:
        self._summary = UsageSummary()
        self._intent_stats = {}


class ServiceRegistry:
    def __init__(self, hook_runner: Optional[HookRunner] = None):
        self._channels: Dict[str, ServiceChannel] = {}
        self._intent_map: Dict[str, ServiceChannel] = {}
        self._keyword_map: Dict[str, List[str]] = {}
        self._hook_runner = hook_runner or HookRunner()

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

    def list_channels(self) -> List[ServiceChannel]:
        return list(self._channels.values())

    def find_channels_by_keyword(self, query: str, limit: int = 20) -> List[ServiceChannel]:
        needle = query.lower()
        matches = set()
        for keyword, channel_names in self._keyword_map.items():
            if needle in keyword.lower():
                for name in channel_names:
                    matches.add(name)
        for name, channel in self._channels.items():
            if needle in name.lower() or needle in channel.description.lower():
                matches.add(name)
        return [self._channels[name] for name in list(matches)[:limit]]

    def execute(
        self,
        intent: str,
        slots: Dict[str, str],
        confidence: float = 0.0,
        session_id: str = "",
        permission_context: Optional[PermissionContext] = None
    ) -> ToolExecution:
        channel = self.get_channel_by_intent(intent)
        if channel is None:
            return ToolExecution(
                channel=ServiceChannel(name='Unknown', intent=intent, description='', keywords=(), required_slots=(), optional_slots=()),
                slots=slots,
                handled=False,
                message=f'未知意图: {intent}',
                confidence=confidence,
                error_code='UNKNOWN_INTENT'
            )

        if permission_context and not permission_context.allows(channel):
            return ToolExecution(
                channel=channel,
                slots=slots,
                handled=False,
                message=f'无权访问此服务通道',
                confidence=confidence,
                error_code='PERMISSION_DENIED'
            )

        hook_context = HookContext(
            session_id=session_id,
            user_input="",
            intent=intent,
            slots=slots,
            channel_name=channel.name
        )
        hook_result = self._hook_runner.run_pre_hooks(hook_context)
        if not hook_result.success:
            return ToolExecution(
                channel=channel,
                slots=slots,
                handled=False,
                message=hook_result.messages[0] if hook_result.messages else '请求被拒绝',
                confidence=confidence,
                error_code='HOOK_DENIED'
            )

        missing_slots = []
        for slot in channel.required_slots:
            if slot not in slots or not slots[slot]:
                missing_slots.append(slot)

        if missing_slots:
            slot_names = {'订单号': '订单号', '手机号': '手机号', '商品名': '商品名称', '收货地址': '新收货地址'}
            missing_str = '、'.join([slot_names.get(s, s) for s in missing_slots])
            return ToolExecution(
                channel=channel,
                slots=slots,
                handled=False,
                message=f'请补充以下信息：{missing_str}',
                confidence=confidence,
                error_code='INCOMPLETE_SLOTS'
            )

        if channel.handler:
            try:
                response = channel.handler(slots=slots, confidence=confidence, session_id=session_id)
                post_hook_result = self._hook_runner.run_post_hooks(hook_context, response)
                if not post_hook_result.success:
                    return ToolExecution(
                        channel=channel,
                        slots=slots,
                        handled=False,
                        message=post_hook_result.messages[0] if post_hook_result.messages else '请求被拒绝',
                        confidence=confidence,
                        error_code='HOOK_DENIED'
                    )
                return ToolExecution(
                    channel=channel,
                    slots=slots,
                    handled=True,
                    message=response,
                    confidence=confidence
                )
            except Exception as e:
                return ToolExecution(
                    channel=channel,
                    slots=slots,
                    handled=False,
                    message=f'处理失败: {str(e)}',
                    confidence=confidence,
                    error_code='HANDLER_ERROR'
                )

        slot_info = '\n'.join([f'{k}: {v}' for k, v in slots.items()])
        return ToolExecution(
            channel=channel,
            slots=slots,
            handled=True,
            message=f'已路由至【{channel.name}】\n\n处理信息:\n{slot_info}',
            confidence=confidence
        )

    def get_channel_stats(self) -> Dict[str, Any]:
        stats = {}
        for name, channel in self._channels.items():
            stats[name] = {
                'intent': channel.intent,
                'description': channel.description,
                'required_slots': len(channel.required_slots),
                'optional_slots': len(channel.optional_slots),
                'keywords': len(channel.keywords),
                'has_handler': channel.handler is not None,
            }
        return stats


service_registry = ServiceRegistry()
usage_tracker = UsageTracker()