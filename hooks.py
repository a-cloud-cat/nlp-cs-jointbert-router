from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any


@dataclass
class HookResult:
    success: bool = True
    messages: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, message: str = "", data: Dict[str, Any] = None) -> "HookResult":
        return cls(
            success=True,
            messages=[message] if message else [],
            data=data or {}
        )

    @classmethod
    def failure(cls, message: str, data: Dict[str, Any] = None) -> "HookResult":
        return cls(
            success=False,
            messages=[message],
            data=data or {}
        )

    def is_denied(self) -> bool:
        return not self.success


@dataclass
class HookContext:
    session_id: str = ""
    user_input: str = ""
    intent: str = ""
    slots: Dict[str, str] = field(default_factory=dict)
    channel_name: str = ""


class HookRunner:
    def __init__(self):
        self._pre_hooks: List[Callable[[HookContext], HookResult]] = []
        self._post_hooks: List[Callable[[HookContext, str], HookResult]] = []

    def register_pre_hook(self, hook: Callable[[HookContext], HookResult]) -> None:
        self._pre_hooks.append(hook)

    def register_post_hook(self, hook: Callable[[HookContext, str], HookResult]) -> None:
        self._post_hooks.append(hook)

    def run_pre_hooks(self, context: HookContext) -> HookResult:
        for hook in self._pre_hooks:
            result = hook(context)
            if not result.success:
                return result
        return HookResult.success()

    def run_post_hooks(self, context: HookContext, response: str) -> HookResult:
        for hook in self._post_hooks:
            result = hook(context, response)
            if not result.success:
                return result
        return HookResult.success()


hook_runner = HookRunner()


def log_request_hook(context: HookContext) -> HookResult:
    print(f"[HOOK] Request: session={context.session_id[:8]}, intent={context.intent}, slots={context.slots}")
    return HookResult.success()


def validate_session_hook(context: HookContext) -> HookResult:
    if not context.session_id:
        return HookResult.failure("session_id is required")
    return HookResult.success()


hook_runner.register_pre_hook(log_request_hook)
hook_runner.register_pre_hook(validate_session_hook)