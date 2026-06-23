from flask import Flask, request, render_template_string, redirect, url_for
import random
import re
import os

app = Flask(__name__)

predictor = None

def init_predictor():
    global predictor
    if predictor is None:
        try:
            from predict import Predictor
            predictor = Predictor()
            print('Predictor initialized successfully')
        except Exception as e:
            print(f'Failed to initialize predictor: {e}')
            predictor = None

INTENT_CHANNEL_MAP = {
    '退货申请': {
        'name': '退货处理通道',
        'description': '处理商品退货、退款相关问题',
        'keywords': ['退货', '退款', '退掉', '不满意', '质量差'],
        'required_slots': ['订单号'],
        'optional_slots': ['商品名']
    },
    '物流查询': {
        'name': '物流查询通道',
        'description': '查询包裹运输状态、预计送达时间',
        'keywords': ['物流', '快递', '到哪了', '包裹', '送达'],
        'required_slots': ['订单号'],
        'optional_slots': ['商品名']
    },
    '改地址': {
        'name': '地址修改通道',
        'description': '修改收货地址、配送地址',
        'keywords': ['改地址', '地址错了', '搬家', '变更地址'],
        'required_slots': ['订单号', '收货地址'],
        'optional_slots': []
    },
    '开发票': {
        'name': '发票服务通道',
        'description': '开具电子发票、纸质发票',
        'keywords': ['发票', '开票', '电子发票', '专票', '普票'],
        'required_slots': ['订单号'],
        'optional_slots': ['商品名']
    },
    '领优惠券': {
        'name': '优惠券领取通道',
        'description': '领取专属优惠券、新人礼包',
        'keywords': ['优惠券', '领券', '折扣', '满减'],
        'required_slots': ['手机号'],
        'optional_slots': []
    },
    '转人工': {
        'name': '人工客服通道',
        'description': '转接专业人工客服一对一服务',
        'keywords': ['人工', '转人工', '找客服'],
        'required_slots': [],
        'optional_slots': ['订单号', '商品名']
    },
    'default': {
        'name': '智能客服通道',
        'description': '其他问题咨询',
        'keywords': [],
        'required_slots': [],
        'optional_slots': []
    }
}

def generate_confirmation(intent, slots, text):
    channel = INTENT_CHANNEL_MAP.get(intent, INTENT_CHANNEL_MAP['default'])
    required_missing = []
    
    for slot in channel['required_slots']:
        if slot not in slots:
            required_missing.append(slot)
    
    if required_missing:
        slot_names = {'订单号': '订单号', '手机号': '手机号', '商品名': '商品名称', '收货地址': '新收货地址'}
        missing_str = '、'.join([slot_names.get(s, s) for s in required_missing])
        return {
            'status': 'incomplete',
            'message': f'检测到您需要【{channel['name']}】服务！\n\n为了帮您快速处理，请补充以下信息：\n{missing_str}',
            'intent': intent,
            'slots': slots,
            'required_missing': required_missing,
            'channel': channel
        }
    
    slot_info = []
    if slots.get('订单号'):
        slot_info.append(f'订单号：{slots['订单号']}')
    if slots.get('商品名'):
        slot_info.append(f'商品名：{slots['商品名']}')
    if slots.get('手机号'):
        slot_info.append(f'手机号：{slots['手机号']}')
    if slots.get('收货地址'):
        slot_info.append(f'收货地址：{slots['收货地址']}')
    
    slot_info_str = '\n'.join(slot_info) if slot_info else '暂无详细信息'
    
    return {
        'status': 'confirmed',
        'message': f'已为您识别需求：\n\n📋 服务类型：【{channel['name']}】\n📝 服务描述：{channel['description']}\n\n已提取信息：\n{slot_info_str}\n\n是否确认以上信息？',
        'intent': intent,
        'slots': slots,
        'required_missing': [],
        'channel': channel
    }

def generate_route_response(intent, slots):
    channel = INTENT_CHANNEL_MAP.get(intent, INTENT_CHANNEL_MAP['default'])
    
    responses = {
        '退货申请': [
            f'好的！已为您接入【退货处理通道】\n\n订单{slots.get('订单号', '')}正在处理中，请耐心等待~',
            f'已为您转接退货专员，订单{slots.get('订单号', '')}的{slots.get('商品名', '')}退货申请已提交~'
        ],
        '物流查询': [
            f'已为您查询物流信息，订单{slots.get('订单号', '')}正在派送中~',
            f'【物流查询通道】已接入，您的{slots.get('商品名', '')}预计今天送达~'
        ],
        '改地址': [
            f'已为您修改地址！订单{slots.get('订单号', '')}将按新地址【{slots.get('收货地址', '')}】配送~',
            f'【地址修改通道】已处理，订单{slots.get('订单号', '')}地址已更新为{slots.get('收货地址', '')}~'
        ],
        '开发票': [
            f'订单{slots.get('订单号', '')}发票已开具！电子发票将发送到您的邮箱~',
            f'【发票服务通道】已处理，{slots.get('商品名', '')}的发票正在开具中~'
        ],
        '领优惠券': [
            f'手机号{slots.get('手机号', '')}已绑定！专属优惠券已发放到您的账户~',
            f'【优惠券领取通道】已处理，恭喜您成功领取优惠券~'
        ],
        '转人工': [
            '好的！正在为您转接人工客服，请稍等片刻~',
            '已接入人工客服通道，专业客服人员将为您提供一对一服务~'
        ],
        'default': [
            '已为您接入智能客服通道，请描述您的问题~',
            '已为您转接至对应服务通道，请继续描述您的需求~'
        ]
    }
    
    return random.choice(responses.get(intent, responses['default']))

@app.route('/', methods=['GET', 'POST'])
def index():
    session = request.args.get('session', 'new')
    result = None
    route_info = None
    
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        confirm_action = request.form.get('confirm_action', '')
        
        if confirm_action == 'confirm':
            intent = request.form.get('intent', 'default')
            slots = {}
            for slot in ['订单号', '手机号', '商品名', '收货地址']:
                val = request.form.get(f'slot_{slot}', '')
                if val:
                    slots[slot] = val
            
            route_info = {
                'intent': intent,
                'slots': slots,
                'channel': INTENT_CHANNEL_MAP.get(intent, INTENT_CHANNEL_MAP['default']),
                'response': generate_route_response(intent, slots)
            }
            
        elif confirm_action == 'reject':
            result = {
                'status': 'rejected',
                'message': '好的！请重新描述您的需求，我会认真倾听~'
            }
            
        elif text:
            if any(keyword in text for keyword in ['转人工', '人工客服', '找人工']):
                prediction = {'intent': '转人工', 'slots': {}}
            else:
                try:
                    init_predictor()
                    if predictor is not None:
                        prediction = predictor.predict(text)
                    else:
                        prediction = {'intent': 'default', 'slots': {}}
                except Exception as e:
                    print(f'Prediction error: {e}')
                    prediction = {'intent': 'default', 'slots': {}}
            
            result = generate_confirmation(prediction['intent'], prediction['slots'], text)
    
    return render_template_string(HTML_TEMPLATE, result=result, route_info=route_info)

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>客服中转中心</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Microsoft YaHei', Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 500px; margin: 0 auto; }
        .card { background: white; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.15); overflow: hidden; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px 25px; text-align: center; }
        .header h1 { font-size: 20px; font-weight: 600; }
        .header p { font-size: 13px; opacity: 0.9; margin-top: 4px; }
        .content { padding: 25px; }
        .message-box { background: #f8f9fa; border-radius: 12px; padding: 16px; margin-bottom: 20px; }
        .message-box p { font-size: 14px; line-height: 1.8; color: #333; white-space: pre-wrap; }
        .status-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; margin-bottom: 10px; }
        .status-incomplete { background: #fff3cd; color: #856404; }
        .status-confirmed { background: #d4edda; color: #155724; }
        .status-rejected { background: #f8d7da; color: #721c24; }
        .status-routed { background: #d1ecf1; color: #0c5460; }
        .quick-buttons { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
        .quick-btn { flex: 1; min-width: 45%; padding: 12px; background: #f0f0f0; border: none; border-radius: 10px; font-size: 14px; color: #333; cursor: pointer; transition: all 0.3s; }
        .quick-btn:hover { background: #667eea; color: white; }
        .input-group { margin-bottom: 15px; }
        .input-group label { display: block; font-size: 13px; color: #666; margin-bottom: 6px; }
        .input-group input { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; outline: none; }
        .input-group input:focus { border-color: #667eea; }
        .action-buttons { display: flex; gap: 12px; margin-top: 20px; }
        .action-btn { flex: 1; padding: 14px; border: none; border-radius: 10px; font-size: 15px; font-weight: 600; cursor: pointer; transition: all 0.3s; }
        .btn-confirm { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .btn-confirm:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102,126,234,0.4); }
        .btn-reject { background: #f0f0f0; color: #666; }
        .btn-reject:hover { background: #e0e0e0; }
        .btn-back { background: #f0f0f0; color: #666; width: 100%; }
        .btn-back:hover { background: #e0e0e0; }
        .chat-input { display: flex; gap: 10px; margin-top: 15px; }
        .chat-input input { flex: 1; padding: 12px 15px; border: 1px solid #ddd; border-radius: 25px; font-size: 14px; outline: none; }
        .chat-input input:focus { border-color: #667eea; }
        .chat-input button { padding: 12px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 25px; font-size: 14px; cursor: pointer; }
        .chat-input button:hover { opacity: 0.9; }
        .channel-info { background: #e8f4fd; border-left: 4px solid #667eea; padding: 12px 15px; border-radius: 0 8px 8px 0; margin-bottom: 15px; }
        .channel-info h3 { font-size: 14px; color: #1a73e8; margin-bottom: 4px; }
        .channel-info p { font-size: 13px; color: #555; }
        .slot-display { background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 10px; }
        .slot-item { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #eee; }
        .slot-item:last-child { border-bottom: none; }
        .slot-label { font-size: 13px; color: #666; }
        .slot-value { font-size: 13px; color: #333; font-weight: 500; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="header">
                <h1>客服中转中心</h1>
                <p>智能识别需求，快速引导至对应服务通道</p>
            </div>
            <div class="content">
                {% if route_info %}
                    <div class="status-badge status-routed">✓ 已路由至服务通道</div>
                    <div class="channel-info">
                        <h3>{{ route_info.channel.name }}</h3>
                        <p>{{ route_info.channel.description }}</p>
                    </div>
                    <div class="message-box">
                        <p>{{ route_info.response }}</p>
                    </div>
                    {% if route_info.slots %}
                        <div class="slot-display">
                            {% for key, value in route_info.slots.items() %}
                                <div class="slot-item">
                                    <span class="slot-label">{{ {'订单号':'订单号', '手机号':'手机号', '商品名':'商品名称', '收货地址':'收货地址'}.get(key, key) }}</span>
                                    <span class="slot-value">{{ value }}</span>
                                </div>
                            {% endfor %}
                        </div>
                    {% endif %}
                    <form method="POST" class="action-buttons">
                        <button type="submit" class="action-btn btn-back">返回重新查询</button>
                    </form>
                {% elif result %}
                    <div class="status-badge {{ 'status-incomplete' if result.status == 'incomplete' else 'status-confirmed' if result.status == 'confirmed' else 'status-rejected' }}">
                        {% if result.status == 'incomplete' %}待补充信息{% elif result.status == 'confirmed' %}信息确认{% else %}已拒绝{% endif %}
                    </div>
                    <div class="message-box">
                        <p>{{ result.message }}</p>
                    </div>
                    {% if result.status == 'incomplete' %}
                        <form method="POST">
                            {% for slot in result.required_missing %}
                                <div class="input-group">
                                    <label>请输入{{ {'订单号':'订单号', '手机号':'手机号', '商品名':'商品名称', '收货地址':'新收货地址'}.get(slot, slot) }}</label>
                                    <input type="text" name="slot_{{ slot }}" placeholder="请输入...">
                                </div>
                            {% endfor %}
                            {% for slot in ['订单号', '手机号', '商品名', '收货地址'] %}
                                {% if slot in result.slots %}
                                    <input type="hidden" name="slot_{{ slot }}" value="{{ result.slots[slot] }}">
                                {% endif %}
                            {% endfor %}
                            <input type="hidden" name="intent" value="{{ result.intent }}">
                            <input type="hidden" name="confirm_action" value="confirm">
                            <button type="submit" class="action-btn btn-confirm">确认并提交</button>
                        </form>
                    {% elif result.status == 'confirmed' %}
                        {% if result.slots %}
                            <div class="slot-display">
                                {% for key, value in result.slots.items() %}
                                    <div class="slot-item">
                                        <span class="slot-label">{{ {'订单号':'订单号', '手机号':'手机号', '商品名':'商品名称', '收货地址':'收货地址'}.get(key, key) }}</span>
                                        <span class="slot-value">{{ value }}</span>
                                    </div>
                                {% endfor %}
                            </div>
                        {% endif %}
                        <form method="POST" class="action-buttons">
                            <input type="hidden" name="intent" value="{{ result.intent }}">
                            {% for slot in ['订单号', '手机号', '商品名', '收货地址'] %}
                                {% if slot in result.slots %}
                                    <input type="hidden" name="slot_{{ slot }}" value="{{ result.slots[slot] }}">
                                {% endif %}
                            {% endfor %}
                            <input type="hidden" name="confirm_action" value="confirm">
                            <button type="submit" class="action-btn btn-confirm">确认无误</button>
                            <input type="hidden" name="confirm_action" value="reject">
                            <button type="submit" class="action-btn btn-reject">重新描述</button>
                        </form>
                    {% else %}
                        <form method="POST" class="chat-input">
                            <input type="text" name="text" placeholder="请重新描述您的需求..." autofocus>
                            <button type="submit">发送</button>
                        </form>
                    {% endif %}
                {% else %}
                    <div class="message-box" style="text-align:center;">
                        <p style="font-size:16px;margin-bottom:10px;">👋 您好！</p>
                        <p>我是客服中转助手，帮您快速找到对应服务通道</p>
                        <p style="font-size:12px;color:#999;margin-top:10px;">请描述您的需求，我会识别并引导您至正确通道</p>
                    </div>
                    <div class="quick-buttons">
                        <button class="quick-btn" onclick="fillInput('我要退货')">📦 退货申请</button>
                        <button class="quick-btn" onclick="fillInput('查物流')">🚚 物流查询</button>
                        <button class="quick-btn" onclick="fillInput('改收货地址')">📍 改地址</button>
                        <button class="quick-btn" onclick="fillInput('开发票')">🧾 开发票</button>
                        <button class="quick-btn" onclick="fillInput('领优惠券')">🎁 领优惠券</button>
                        <button class="quick-btn" onclick="fillInput('转人工')">👤 转人工</button>
                    </div>
                    <form method="POST" class="chat-input">
                        <input type="text" name="text" id="inputText" placeholder="请输入您的需求..." onkeydown="if(event.keyCode==13) this.form.submit()">
                        <button type="submit">发送</button>
                    </form>
                {% endif %}
            </div>
        </div>
    </div>
    <script>
        function fillInput(text) {
            document.getElementById('inputText').value = text;
            document.querySelector('form').submit();
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    print('=' * 50)
    print('客服中转中心已启动！')
    print('访问地址: http://127.0.0.1:5000')
    print('=' * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)