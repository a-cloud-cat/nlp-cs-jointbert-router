from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Message, MessageRole, Session


class SessionService:
    def __init__(self, max_sessions: int = 1000, ttl_seconds: int = 3600, max_messages: int = 30):
        self._sessions: Dict[str, Session] = {}
        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds
        self._max_messages = max_messages

    def create_session(self, session_id: Optional[str] = None) -> Session:
        if len(self._sessions) >= self._max_sessions:
            self._cleanup_expired()

        session_id = session_id or str(uuid.uuid4())
        session = Session(session_id=session_id)
        session.add_message(MessageRole.ASSISTANT, "亲，您有什么需求？")
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        session = self._sessions.get(session_id)
        if session and self._is_expired(session):
            del self._sessions[session_id]
            return None
        return session

    def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        if session_id and self.get_session(session_id):
            return self._sessions[session_id]
        return self.create_session(session_id)

    def add_message(self, session_id: str, role: MessageRole, text: str) -> None:
        session = self.get_session(session_id)
        if session:
            session.add_message(role, text)
            self._auto_compact(session)

    def get_messages(self, session_id: str) -> list:
        session = self.get_session(session_id)
        if not session:
            return []
        return [
            {'role': m.role.value, 'text': m.text}
            for m in session.messages
        ]

    def get_user_message_count(self, session_id: str) -> int:
        session = self.get_session(session_id)
        if not session:
            return 0
        return session.get_user_message_count()

    def get_conversation_summary(self, session_id: str) -> str:
        session = self.get_session(session_id)
        if not session:
            return ""
        return session.get_conversation_summary()

    def update_slots(self, session_id: str, new_slots: Dict[str, str]) -> None:
        session = self.get_session(session_id)
        if session:
            session.update_slots(new_slots)

    def append_intent(self, session_id: str, intent: str, confidence: float) -> None:
        session = self.get_session(session_id)
        if session:
            session.append_intent(intent, confidence)

    def get_slots(self, session_id: str) -> Dict[str, str]:
        session = self.get_session(session_id)
        if not session:
            return {}
        return session.get_slots()

    def get_metadata(self, session_id: str, key: str, default=None):
        session = self.get_session(session_id)
        if not session:
            return default
        return session.metadata.get(key, default)

    def set_metadata(self, session_id: str, key: str, value) -> None:
        session = self.get_session(session_id)
        if session:
            session.metadata[key] = value

    def compact_session(self, session_id: str, keep_messages: int = 10) -> None:
        session = self.get_session(session_id)
        if not session:
            return

        if len(session.messages) <= keep_messages:
            return

        early_messages = session.messages[:-keep_messages]
        summary = self._generate_compact_summary(early_messages, session)

        session.messages = [
            Message(role=MessageRole.ASSISTANT, text=f"[对话已压缩] {summary}"),
            *session.messages[-keep_messages:]
        ]
        session.mark_compacted()

    def _auto_compact(self, session: Session) -> None:
        if len(session.messages) > self._max_messages:
            self.compact_session(session.session_id)

    def _generate_compact_summary(self, messages: list, session: Session) -> str:
        user_texts = [m.text for m in messages if m.role == MessageRole.USER]
        slots = session.get_slots()
        intent_history = session.metadata.get('intent_history', [])

        summary_parts = []

        if slots:
            slot_str = ", ".join(f"{k}: {v}" for k, v in slots.items())
            summary_parts.append(f"关键信息: {slot_str}")

        if intent_history:
            recent_intents = intent_history[-3:] if len(intent_history) > 3 else intent_history
            intent_str = " → ".join(i['intent'] for i in recent_intents)
            summary_parts.append(f"意图: {intent_str}")

        if user_texts:
            recent_texts = user_texts[-2:] if len(user_texts) > 2 else user_texts
            summary_parts.append(f"对话要点: {'；'.join(recent_texts)}")

        return " | ".join(summary_parts) if summary_parts else "用户咨询相关问题"

    def _is_expired(self, session: Session) -> bool:
        elapsed = (datetime.now() - session.last_updated).total_seconds()
        return elapsed > self._ttl_seconds

    def _cleanup_expired(self) -> None:
        expired_ids = [
            session_id for session_id, session in self._sessions.items()
            if self._is_expired(session)
        ]
        for session_id in expired_ids:
            del self._sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]

    def reset_all_sessions(self) -> None:
        self._sessions.clear()

    def get_stats(self) -> dict:
        compacted_count = sum(1 for s in self._sessions.values() if s.is_compacted())
        return {
            'active_sessions': len(self._sessions),
            'max_sessions': self._max_sessions,
            'ttl_seconds': self._ttl_seconds,
            'compacted_sessions': compacted_count,
        }


_session_service: Optional[SessionService] = None


def get_session_service() -> SessionService:
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service