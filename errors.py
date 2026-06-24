from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any


class AppError(Exception):
    def __init__(self, code: str, message: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class IntentError(AppError):
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__("INTENT_ERROR", message, details)


class SlotError(AppError):
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__("SLOT_ERROR", message, details)


class ServiceError(AppError):
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__("SERVICE_ERROR", message, details)


class PermissionError(AppError):
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__("PERMISSION_ERROR", message, details)


class SessionError(AppError):
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__("SESSION_ERROR", message, details)


class ValidationError(AppError):
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__("VALIDATION_ERROR", message, details)


@dataclass
class ErrorResponse:
    success: bool = False
    error_code: str = ""
    error_message: str = ""
    error_details: Dict[str, Any] = None

    @classmethod
    def from_error(cls, error: AppError) -> "ErrorResponse":
        return cls(
            success=False,
            error_code=error.code,
            error_message=error.message,
            error_details=error.details,
        )

    @classmethod
    def success_response(cls) -> "ErrorResponse":
        return cls(success=True)

    def to_dict(self) -> Dict[str, Any]:
        if self.success:
            return {"success": True}
        return {
            "success": False,
            "error": {
                "code": self.error_code,
                "message": self.error_message,
                "details": self.error_details or {},
            },
        }