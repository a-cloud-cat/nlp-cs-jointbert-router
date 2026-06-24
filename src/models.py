from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChannelGroup(Enum):
    TRANSACTION_AFTER_SALES = "交易售后类"
    LOGISTICS_DELIVERY = "物流配送类"
    INVOICE_DISCOUNT = "票据优惠类"
    FALLBACK = "兜底通道"


@dataclass(frozen=True)
class ServiceChannel:
    name: str
    intent: str
    group: ChannelGroup
    api: str
    method: str
    description: str
    keywords: Tuple[str, ...] = ()
    required_slots: Tuple[str, ...] = ()
    optional_slots: Tuple[str, ...] = ()

    def has_keyword(self, text: str) -> bool:
        return any(keyword in text for keyword in self.keywords)


@dataclass(frozen=True)
class Message:
    role: MessageRole
    text: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Session:
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, object] = field(default_factory=dict)

    def add_message(self, role: MessageRole, text: str) -> None:
        self.messages.append(Message(role=role, text=text))
        self.last_updated = datetime.now()

    def get_user_message_count(self) -> int:
        return len([m for m in self.messages if m.role == MessageRole.USER])

    def get_conversation_summary(self) -> str:
        return " ".join(m.text for m in self.messages if m.role == MessageRole.USER)


@dataclass(frozen=True)
class Prediction:
    intent: str
    slots: Dict[str, str]
    confidence: float
    group: ChannelGroup
    clarification_needed: bool
    clarification_question: str


@dataclass
class ReviewItem:
    id: str
    session_id: str
    conversation_summary: str
    full_history: str
    intent: str
    confidence: float
    slots: Dict[str, str]
    group: ChannelGroup
    rounds: int
    created_at: datetime
    status: str = "pending"


@dataclass(frozen=True)
class RouteResult:
    channel: ServiceChannel
    slots: Dict[str, str]
    success: bool
    message: str
    confidence: float = 0.0
    error_code: str = ""


@dataclass(frozen=True)
class UsageStats:
    total_requests: int = 0
    successful_routes: int = 0
    failed_routes: int = 0
    avg_confidence: float = 0.0