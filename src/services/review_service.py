from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ChannelGroup, ReviewItem


class ReviewService:
    def __init__(self):
        self._pending_reviews: List[ReviewItem] = []

    def add_review(
        self,
        session_id: str,
        conversation_summary: str,
        full_history: str,
        intent: str,
        confidence: float,
        slots: Dict[str, str],
        group: ChannelGroup,
        rounds: int,
    ) -> ReviewItem:
        formatted_history = self._format_history(full_history)

        review_item = ReviewItem(
            id=str(uuid.uuid4()),
            session_id=session_id,
            conversation_summary=conversation_summary,
            full_history=formatted_history,
            intent=intent,
            confidence=confidence,
            slots=slots,
            group=group,
            rounds=rounds,
            created_at=datetime.now(),
            status="pending",
        )
        self._pending_reviews.append(review_item)
        return review_item

    def _format_history(self, full_history: str) -> str:
        try:
            import ast
            data = ast.literal_eval(full_history)
            if isinstance(data, list):
                lines = []
                for msg in data:
                    if isinstance(msg, dict):
                        role = msg.get('role', '')
                        text = msg.get('text', '')
                        if role == 'user':
                            lines.append(f'用户: {text}')
                        elif role == 'assistant':
                            lines.append(f'客服: {text}')
                return '\n'.join(lines) if lines else full_history
        except Exception:
            pass
        return full_history

    def get_pending_reviews(self) -> List[ReviewItem]:
        return [r for r in self._pending_reviews if r.status == "pending"]

    def get_review(self, review_id: str) -> Optional[ReviewItem]:
        return next((r for r in self._pending_reviews if r.id == review_id), None)

    def approve_review(self, review_id: str) -> bool:
        review = self.get_review(review_id)
        if review:
            review.status = "approved"
            return True
        return False

    def reject_review(self, review_id: str) -> bool:
        review = self.get_review(review_id)
        if review:
            review.status = "rejected"
            return True
        return False

    def reclassify_review(self, review_id: str, new_intent: str, new_group: ChannelGroup) -> bool:
        review = self.get_review(review_id)
        if review:
            review.intent = new_intent
            review.group = new_group
            return True
        return False

    def remove_review(self, review_id: str) -> bool:
        original_length = len(self._pending_reviews)
        self._pending_reviews = [r for r in self._pending_reviews if r.id != review_id]
        return len(self._pending_reviews) < original_length

    def get_stats(self) -> dict:
        pending = len(self.get_pending_reviews())
        approved = len([r for r in self._pending_reviews if r.status == "approved"])
        rejected = len([r for r in self._pending_reviews if r.status == "rejected"])
        return {
            'pending': pending,
            'approved': approved,
            'rejected': rejected,
            'total': len(self._pending_reviews),
        }


_review_service: Optional[ReviewService] = None


def get_review_service() -> ReviewService:
    global _review_service
    if _review_service is None:
        _review_service = ReviewService()
    return _review_service