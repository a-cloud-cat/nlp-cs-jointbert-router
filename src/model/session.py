from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentBlockType(str, Enum):
    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


@dataclass(frozen=True)
class TextBlock:
    type: ContentBlockType = ContentBlockType.TEXT
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "text": self.text}


@dataclass(frozen=True)
class ToolUseBlock:
    type: ContentBlockType = ContentBlockType.TOOL_USE
    id: str = ""
    name: str = ""
    input: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "id": self.id, "name": self.name, "input": self.input}


@dataclass(frozen=True)
class ToolResultBlock:
    type: ContentBlockType = ContentBlockType.TOOL_RESULT
    tool_use_id: str = ""
    tool_name: str = ""
    output: str = ""
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "tool_use_id": self.tool_use_id,
            "tool_name": self.tool_name,
            "output": self.output,
            "is_error": self.is_error,
        }


@dataclass
class ConversationMessage:
    role: MessageRole
    blocks: List[Any]
    timestamp: datetime = field(default_factory=datetime.now)
    usage: Optional[Dict[str, int]] = None

    @classmethod
    def user_text(cls, text: str) -> "ConversationMessage":
        return cls(
            role=MessageRole.USER,
            blocks=[TextBlock(text=text)],
            timestamp=datetime.now(),
        )

    @classmethod
    def assistant_text(cls, text: str) -> "ConversationMessage":
        return cls(
            role=MessageRole.ASSISTANT,
            blocks=[TextBlock(text=text)],
            timestamp=datetime.now(),
        )

    @classmethod
    def tool_use(cls, tool_id: str, tool_name: str, tool_input: str) -> "ConversationMessage":
        return cls(
            role=MessageRole.TOOL,
            blocks=[ToolUseBlock(id=tool_id, name=tool_name, input=tool_input)],
            timestamp=datetime.now(),
        )

    @classmethod
    def tool_result(cls, tool_use_id: str, tool_name: str, output: str, is_error: bool = False) -> "ConversationMessage":
        return cls(
            role=MessageRole.TOOL,
            blocks=[ToolResultBlock(tool_use_id=tool_use_id, tool_name=tool_name, output=output, is_error=is_error)],
            timestamp=datetime.now(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "blocks": [block.to_dict() for block in self.blocks],
            "timestamp": self.timestamp.isoformat(),
            "usage": self.usage,
        }


@dataclass
class Session:
    session_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    clarification_rounds: int = 0
    current_intent: Optional[str] = None
    current_confidence: float = 0.0

    def add_message(self, message: ConversationMessage) -> None:
        self.messages.append(message)
        self.last_updated = datetime.now()
        if message.role == MessageRole.USER:
            self.clarification_rounds += 1

    def get_user_messages(self) -> List[ConversationMessage]:
        return [m for m in self.messages if m.role == MessageRole.USER]

    def get_latest_user_message(self) -> Optional[str]:
        user_messages = self.get_user_messages()
        if user_messages:
            last_msg = user_messages[-1]
            for block in last_msg.blocks:
                if isinstance(block, TextBlock):
                    return block.text
        return None

    def get_history_text(self, limit: int = 5) -> str:
        recent = self.messages[-limit * 2 :]
        lines = []
        for msg in recent:
            role_prefix = {"system": "[系统]", "user": "[用户]", "assistant": "[助手]", "tool": "[工具]"}.get(msg.role.value, "")
            for block in msg.blocks:
                if isinstance(block, TextBlock):
                    lines.append(f"{role_prefix} {block.text}")
                elif isinstance(block, ToolUseBlock):
                    lines.append(f"{role_prefix} 调用工具: {block.name}")
                elif isinstance(block, ToolResultBlock):
                    lines.append(f"{role_prefix} 工具结果: {block.output[:50]}...")
        return "\n".join(lines)

    def update_intent(self, intent: str, confidence: float) -> None:
        self.current_intent = intent
        self.current_confidence = confidence

    def is_exceeded_rounds(self, max_rounds: int = 5) -> bool:
        return self.clarification_rounds >= max_rounds

    def get_conversation_summary(self) -> str:
        user_texts = []
        for msg in self.get_user_messages():
            for block in msg.blocks:
                if isinstance(block, TextBlock):
                    user_texts.append(block.text)
        return " ".join(user_texts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "message_count": len(self.messages),
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata,
        }


class SessionManager:
    def __init__(self, max_sessions: int = 1000, ttl_seconds: int = 3600):
        self._sessions: Dict[str, Session] = {}
        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds

    def create_session(self, session_id: str) -> Session:
        if len(self._sessions) >= self._max_sessions:
            self._cleanup_expired()

        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        session = self._sessions.get(session_id)
        if session and self._is_expired(session):
            del self._sessions[session_id]
            return None
        return session

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def _is_expired(self, session: Session) -> bool:
        elapsed = (datetime.now() - session.last_updated).total_seconds()
        return elapsed > self._ttl_seconds

    def _cleanup_expired(self) -> None:
        expired_ids = [sid for sid, sess in self._sessions.items() if self._is_expired(sess)]
        for sid in expired_ids:
            del self._sessions[sid]

    def get_stats(self) -> Dict[str, int]:
        return {
            "active_sessions": len(self._sessions),
            "total_messages": sum(len(s.messages) for s in self._sessions.values()),
        }


session_manager = SessionManager()