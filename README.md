# 客服中转中心（Service Router）

电商客服智能中转路由系统，基于 PyTorch JointBERT 模型实现意图识别与槽位填充，支持多轮对话澄清、服务渠道路由、人工审核确认等功能。

## 功能特点

- **NLP 能力**：规则匹配 + BERT 模型双轨意图识别，jieba 中文分词
- **意图分类**：覆盖交易售后（10类）、物流配送（7类）、票据优惠（6类）全场景
- **槽位提取**：自动识别订单号、手机号、商品名、收货地址、发票抬头等关键信息
- **置信度判定**：高置信直接跳转、中置信追问澄清、低置信转人工
- **多轮对话**：最多 5 轮对话澄清，避免单句断章取义
- **人工审核**：支持开启人工复核模式，管理员确认后才执行分发
- **API 预留**：完整的 API 转发配置页面，便于后续对接真实接口

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端层                               │
│   用户输入 Tab | 管理员审核 Tab | API 转发 Tab              │
│   (防重复提交按钮状态管理)                                  │
├─────────────────────────────────────────────────────────────┤
│                        应用层                               │
│                    app.py (Flask)                           │
│   - 路由处理 / 会话管理 / 审核流程                           │
├─────────────────────────────────────────────────────────────┤
│                        业务层                               │
│   NLPService | SessionService | ReviewService               │
│   (意图识别 / 会话管理 / 审核管理)                           │
├─────────────────────────────────────────────────────────────┤
│                        核心层                               │
│   ServiceRegistry | Models | Predictor                     │
│   (服务注册 / 数据模型 / 预测引擎)                           │
├─────────────────────────────────────────────────────────────┤
│                        模型层                               │
│   JointBERT (BERT + 意图分类 + CRF 槽位标注)                │
└─────────────────────────────────────────────────────────────┘
```

## 项目结构

```
zryy/
├── src/                              # 源代码目录
│   ├── __init__.py                   # 模块导出
│   ├── models.py                     # 数据模型定义
│   ├── registry.py                   # 服务注册中心
│   └── services/                     # 业务服务层
│       ├── nlp_service.py            # NLP 意图识别与槽位提取
│       ├── session_service.py         # 会话创建与消息管理
│       └── review_service.py          # 人工审核流程管理
├── app.py                            # Flask 应用入口
├── predict.py                         # 模型预测接口
├── model.py                           # JointBERT 模型架构
├── train.py                           # 模型训练脚本
├── data_process.py                    # 数据预处理（Jieba分词、BIO标签）
├── data/                              # 训练数据集
├── checkpoints/                       # 模型检查点
├── requirements.txt                   # 依赖清单
└── README.md                          # 项目说明
```

## 核心模块

### src/models.py

定义系统核心数据模型（采用现代 Python 架构风格，使用 frozen dataclass 确保数据不可变、可变 dataclass 支持状态流转）：

| 模型 | 说明 |
|------|------|
| `MessageRole` | 消息角色枚举（USER / ASSISTANT） |
| `ChannelGroup` | 渠道分组枚举（交易售后/物流配送/票据优惠/兜底） |
| `ServiceChannel` | 服务通道定义（frozen dataclass） |
| `Session` | 会话管理 |
| `Message` | 消息结构 |
| `Prediction` | 意图预测结果（含置信度、槽位、澄清问题） |
| `ReviewItem` | 审核项（可变 dataclass，支持状态变更） |
| `AppError` | 异常基类 |

### src/registry.py

服务注册中心（采用注册器模式，统一管理服务通道的注册、查询与路由）：

- `register()` 注册服务通道及其关键词映射
- `get_channel()` / `get_channel_by_intent()` 按名称/意图查询
- `find_channels_by_keyword()` 关键词匹配路由
- `get_channels_by_group()` 按分组查询

### src/services/nlp_service.py

NLP 服务模块：

- **意图识别**：规则匹配 + 模型预测双轨融合
- **槽位提取**：正则匹配订单号、手机号、地址等
- **澄清问题**：每种意图预置 3 条多样化问题，随机选取
- **置信度判定**：高(≥0.85) / 中(≥0.4) / 低 三档

### src/services/session_service.py

会话服务模块：

- 会话创建与持久化管理
- 消息添加与历史查询
- 用户消息计数（用于控制对话轮次）
- 多轮对话汇总（用于最终意图识别）

### src/services/review_service.py

审核服务模块：

- 待审核列表管理（添加、移除、查询）
- 审核操作（确认分发、修改意图分类）
- 聊天记录格式化展示
- 审核统计

## 架构优化亮点

本项目参考 C 模型的设计方案，对代码结构进行了以下优化：

### 1. 注册器模式（Registry Pattern）

`ServiceRegistry` 采用中心化注册机制，所有服务通道通过 `register()` 统一注册，支持：
- 按名称/意图快速查询
- 关键词匹配路由
- 按分组管理服务

### 2. 不可变数据模型（Immutable Data）

核心数据对象使用 `frozen dataclass` 定义，确保数据不可变：
- `ServiceChannel`、`Prediction` 等对象创建后状态不会被意外修改
- 仅状态流转对象（如 `ReviewItem`）使用可变 dataclass

### 3. 服务层分离（Service Layer）

业务逻辑按功能拆分为独立服务模块：
- `NLPService`：意图识别与槽位提取
- `SessionService`：会话管理
- `ReviewService`：审核流程
- 各服务职责单一，易于测试和维护

### 4. 查询引擎模式（Query Engine）

`NLPService` 实现了双轨意图识别机制：
- 规则匹配：快速关键词匹配
- 模型预测：基于 BERT 的深度学习分类
- 结果融合：取置信度更高的结果

### 5. 结构化异常处理

定义了完整的异常体系：
- `AppError` 作为基类
- 各业务异常继承自基类
- 便于统一错误处理和日志记录

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python app.py
```

访问地址：http://127.0.0.1:5000

### 3. 训练模型（可选）

```bash
python train.py
```

## 界面说明

### 用户输入 Tab

- 智能客服聊天界面，机器人主动问候"亲，您有什么需求？"
- 用户输入需求后，AI 根据置信度进行追问或直接确认
- 显示剩余对话轮次（最多 5 轮）
- 第 5 轮后总结并跳转到对应渠道

### 管理员审核 Tab

- 待审核列表展示（会话摘要、置信度、对话轮次）
- 完整聊天记录查看（格式化显示）
- 槽位信息标签展示
- 操作按钮：**确认分发**（含意图选择下拉框）

### API 转发 Tab

- 所有服务渠道的 API 配置信息
- 按分组展示（交易售后类 / 物流配送类 / 票据优惠类 / 兜底通道）
- 显示渠道名称、请求方式、API 地址、描述

## 意图分类体系

### 交易售后类（10个）

退货申请、退款申请、换货申请、拒收、售后维权、商品投诉、商家投诉、质量问题、货不对板、少发漏发

### 物流配送类（7个）

物流查询、查快递、发货时效、丢件破损、改地址、改手机号、改配送

### 票据优惠类（6个）

开发票、改发票抬头、补开发票、领优惠券、店铺券、价格补差

### 兜底通道（2个）

转人工、default（未知意图兜底）

## 置信度判定机制

| 置信度 | 处理方式 |
|--------|---------|
| ≥ 0.85 | 高置信，AI 直接确认并跳转渠道 |
| 0.4 ~ 0.85 | 中置信，AI 主动追问澄清 |
| < 0.4 | 低置信，自动转人工 |

## 全局审核开关

- **关闭审核（默认）**：AI 识别后直接跳转对应渠道
- **开启审核**：识别结果推入管理员审核列表，需人工确认分发

## 扩展指南

### 添加新服务渠道

在 `src/registry.py` 的 `_build_default_registry()` 函数中添加：

```python
registry.register(ServiceChannel(
    name='新渠道名称',
    intent='新意图',
    group=ChannelGroup.TRANSACTION_AFTER_SALES,
    api='/api/services/new_channel',
    method='POST',
    description='描述信息',
    keywords=('关键词1', '关键词2'),
    required_slots=('订单号',),
))
```

### 添加澄清问题

在 `src/services/nlp_service.py` 的 `CLARIFICATION_QUESTIONS` 字典中添加：

```python
'新意图': ['问题1', '问题2', '问题3'],
```

## 技术栈

- Python 3.9+
- Flask 2.0+
- PyTorch
- Transformers (BERT)
- jieba（中文分词）

## License

MIT License
