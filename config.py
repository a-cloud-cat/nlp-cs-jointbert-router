import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    model_name: str = "bert-base-chinese"
    model_path: str = "./models/bert-base-chinese"
    checkpoint_path: str = "./checkpoints/best_model.pt"
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 10
    patience: int = 3
    max_seq_len: int = 128


@dataclass
class IntentConfig:
    labels: List[str] = field(default_factory=lambda: [
        "退货申请", "物流查询", "改地址", "开发票", "领优惠券", "转人工", "default"
    ])
    rule_confidence_threshold: float = 0.7


@dataclass
class SlotConfig:
    labels: List[str] = field(default_factory=lambda: [
        "O", "B-订单号", "I-订单号", "B-手机号", "I-手机号",
        "B-商品名", "I-商品名", "B-收货地址", "I-收货地址"
    ])


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    session_ttl: int = 3600
    max_sessions: int = 1000


@dataclass
class LogConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    slot: SlotConfig = field(default_factory=SlotConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    log: LogConfig = field(default_factory=LogConfig)


_config = AppConfig()


def load_config() -> AppConfig:
    return _config


def get_model_config() -> ModelConfig:
    return _config.model


def get_intent_config() -> IntentConfig:
    return _config.intent


def get_slot_config() -> SlotConfig:
    return _config.slot


def get_server_config() -> ServerConfig:
    return _config.server


def get_log_config() -> LogConfig:
    return _config.log