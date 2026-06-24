from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, Optional

from ..models import Message, MessageRole, Session


class SessionService:
    def __init__(self, max_sessions: int = 1000, ttl_seconds: int = 3600):
        self._sessions: Dict[str, Session] = {}
        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds

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

    def get_stats(self) -> dict:
        return {
            'active_sessions': len(self._sessions),
            'max_sessions': self._max_sessions,
            'ttl_seconds': self._ttl_seconds,
        }


_session_service: Optional[SessionService] = None


def get_session_service() -> SessionService:
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service