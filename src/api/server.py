import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, render_template_string, session as flask_session, jsonify

from models import ChannelGroup, MessageRole
from registry import get_registry
from services.nlp_service import get_nlp_service, MAX_CLARIFICATION_ROUNDS
from services.session_service import get_session_service
from services.review_service import get_review_service

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

enable_human_review = False
enable_long_conversation = False
LONG_CONVERSATION_ROUNDS = 50


@app.route('/', methods=['GET', 'POST'])
def index():
    global enable_human_review, enable_long_conversation

    session_service = get_session_service()
    nlp_service = get_nlp_service()
    review_service = get_review_service()
    registry = get_registry()

    current_view = flask_session.get('current_view', 'user')
    session_id = flask_session.get('session_id')

    if not session_id:
        session_id = str(id(flask_session))
        flask_session['session_id'] = session_id

    session_service.get_or_create_session(session_id)
    messages = session_service.get_messages(session_id)
    user_message_count = session_service.get_user_message_count(session_id)
    
    max_rounds = LONG_CONVERSATION_ROUNDS if enable_long_conversation else MAX_CLARIFICATION_ROUNDS
    remaining_rounds = max(0, max_rounds - user_message_count)

    show_redirect_modal = False
    redirect_channel = None
    redirect_api = None
    show_success_modal = False
    success_message = ''

    if request.method == 'POST':
        view_switch = request.form.get('view_switch', '')
        review_switch = request.form.get('review_switch', '')
        user_text = request.form.get('user_text', '').strip()
        admin_action = request.form.get('admin_action', '')

        if review_switch == 'toggle':
            enable_human_review = not enable_human_review
            return jsonify({'success': True, 'enable_human_review': enable_human_review})
        
        if request.form.get('long_conversation_switch') == 'toggle':
            enable_long_conversation = not enable_long_conversation
            return jsonify({'success': True, 'enable_long_conversation': enable_long_conversation})

        if view_switch:
            flask_session['current_view'] = view_switch
            current_view = view_switch

        elif current_view == 'user':
                if user_text:
                    session_service.add_message(session_id, MessageRole.USER, user_text)

                    if any(keyword in user_text for keyword in ['转人工', '人工客服', '找人工']):
                        session_service.add_message(session_id, MessageRole.ASSISTANT, '好的，正在为您转接人工客服，请稍候...')
                        show_redirect_modal = True
                        redirect_channel = '转人工'
                        redirect_api = registry.get_channel_by_intent('转人工').api
                    else:
                        user_count = session_service.get_user_message_count(session_id)
                        max_rounds = LONG_CONVERSATION_ROUNDS if enable_long_conversation else MAX_CLARIFICATION_ROUNDS

                        if user_count >= max_rounds:
                            conversation_summary = session_service.get_conversation_summary(session_id)
                            prediction = nlp_service.predict(conversation_summary, session_id)

                            intent = prediction.intent
                            if intent == 'default':
                                intent = '转人工'

                            slots_info = session_service.get_slots(session_id)
                            slots_str = '\n'.join(f'· {k}: {v}' for k, v in slots_info.items()) if slots_info else '暂无'

                            session_service.add_message(session_id, MessageRole.ASSISTANT,
                                f'根据您的描述，我已经了解您的问题了。\n\n【总结】\n您的问题属于：{prediction.group.value}\n服务类型：{intent}\n\n【已识别信息】\n{slots_str}\n\n正在为您跳转到对应服务渠道...')

                            if enable_human_review:
                                review_service.add_review(
                                    session_id=session_id,
                                    conversation_summary=conversation_summary,
                                    full_history=str(messages),
                                    intent=intent,
                                    confidence=prediction.confidence,
                                    slots=slots_info,
                                    group=prediction.group,
                                    rounds=user_count,
                                )
                            else:
                                show_redirect_modal = True
                                redirect_channel = intent
                                channel = registry.get_channel_by_intent(intent)
                                redirect_api = channel.api if channel else ''
                        else:
                            is_learning = session_service.get_metadata(session_id, 'is_learning_product', False)

                            if is_learning:
                                learned_product = nlp_service.learn_product(user_text)
                                if learned_product:
                                    session_service.set_metadata(session_id, 'is_learning_product', False)
                                    session_service.update_slots(session_id, {'商品名': learned_product})
                                    session_service.add_message(session_id, MessageRole.ASSISTANT,
                                        f'好的，我已经记住了「{learned_product}」。\n\n请问您想对「{learned_product}」进行什么操作呢？（退货、退款、换货、开发票等）')
                                else:
                                    session_service.add_message(session_id, MessageRole.ASSISTANT,
                                        '抱歉，我没能理解您说的商品名称。请直接告诉我商品名称，例如"水杯"、"柜子"。')
                            else:
                                prev_intent = session_service.get_metadata(session_id, 'last_intent', None)
                                prev_slots = session_service.get_slots(session_id)

                                prediction = nlp_service.predict(user_text, session_id)
                                max_rounds = LONG_CONVERSATION_ROUNDS if enable_long_conversation else MAX_CLARIFICATION_ROUNDS
                                remaining = max(0, max_rounds - user_count)
                                slots_info = session_service.get_slots(session_id)

                                need_clarify_product = False
                                if prediction.intent in ['退货申请', '退款申请', '换货申请', '质量问题', '货不对板', '开发票', '物流查询', '查快递', '商品投诉']:
                                    product_name = nlp_service.extract_product(user_text)
                                    if not product_name:
                                        if slots_info and '商品名' in slots_info:
                                            product_name = slots_info['商品名']
                                        else:
                                            need_clarify_product = True

                                context_keywords = ['质量', '发霉', '坏了', '甲醛', '异味', '破损', '投诉', '退货', '退款', '换货']
                                has_context_keyword = any(kw in user_text for kw in context_keywords)

                                if prev_intent and prev_intent != 'default' and prediction.intent == 'default' and has_context_keyword:
                                    prediction = nlp_service.predict(f'{prev_intent}: {user_text}', session_id)

                                if prediction.intent != 'default':
                                    session_service.set_metadata(session_id, 'last_intent', prediction.intent)

                                if need_clarify_product:
                                    session_service.set_metadata(session_id, 'is_learning_product', True)
                                    clarification_text = '抱歉，我没能识别出您提到的商品。\n\n请告诉我您说的是什么商品呢？例如"水杯"、"柜子"。'
                                elif prev_intent and prev_intent != 'default' and prediction.intent == 'default':
                                    clarification_text = '我理解您遇到了问题，请告诉我具体是什么情况，我会尽力帮您处理。'
                                elif slots_info:
                                    slots_str = '，'.join(f'{k}={v}' for k, v in slots_info.items())
                                    clarification_text = f'{prediction.clarification_question}\n\n【已识别信息】{slots_str}'
                                else:
                                    clarification_text = prediction.clarification_question

                                session_service.add_message(session_id, MessageRole.ASSISTANT, clarification_text)

        elif current_view == 'admin':
            if admin_action == 'submit':
                try:
                    review_id = request.form.get('review_id')
                    final_intent = request.form.get('final_intent')
                    print(f'[DEBUG] submit: review_id={review_id}, final_intent={final_intent}')
                    channel = registry.get_channel_by_intent(final_intent)
                    print(f'[DEBUG] channel={channel}')
                    if channel:
                        review_service.reclassify_review(review_id, final_intent, channel.group)
                    review_service.approve_review(review_id)
                    review_service.remove_review(review_id)
                    show_success_modal = True
                    success_message = f'提交成功！已分发到 {final_intent}'
                except Exception as e:
                    print(f'[ERROR] submit failed: {e}')
                    import traceback
                    traceback.print_exc()

    messages = session_service.get_messages(session_id)
    user_message_count = session_service.get_user_message_count(session_id)
    max_rounds = LONG_CONVERSATION_ROUNDS if enable_long_conversation else MAX_CLARIFICATION_ROUNDS
    remaining_rounds = max(0, max_rounds - user_message_count)

    pending_reviews = review_service.get_pending_reviews()

    admin_view = {
        'pending_reviews': pending_reviews,
        'total_pending': len(pending_reviews),
        'intents': registry.list_all_intents()
    }

    channels = registry.list_all_channels()
    groups = sorted(set(c.group for c in channels), key=lambda g: g.value)

    api_view = {
        'channels': channels,
        'groups': groups
    }

    return render_template_string(HTML_TEMPLATE,
        view=current_view,
        messages=messages,
        user_message_count=user_message_count,
        remaining_rounds=remaining_rounds,
        admin_view=admin_view,
        api_view=api_view,
        enable_human_review=enable_human_review,
        enable_long_conversation=enable_long_conversation,
        show_redirect_modal=show_redirect_modal,
        redirect_channel=redirect_channel,
        redirect_api=redirect_api,
        show_success_modal=show_success_modal,
        success_message=success_message,
        channel_group_enum=ChannelGroup
    )


@app.route('/reset', methods=['POST'])
def reset_conversation():
    import uuid
    
    session_service = get_session_service()
    session_id = flask_session.get('session_id')
    
    if session_id:
        session_service.delete_session(session_id)
    
    flask_session.clear()
    flask_session.permanent = False
    
    new_session_id = str(uuid.uuid4())
    flask_session['session_id'] = new_session_id
    
    return jsonify({'success': True, 'message': '对话已重置', 'new_session_id': new_session_id})


@app.route('/api/health')
def health():
    session_service = get_session_service()
    review_service = get_review_service()
    return jsonify({
        'status': 'ok',
        'session_stats': session_service.get_stats(),
        'review_stats': review_service.get_stats(),
        'enable_human_review': enable_human_review,
        'enable_long_conversation': enable_long_conversation
    })


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>客服中转中心</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Microsoft YaHei', sans-serif; background: #f5f5f5; color: #333; }
        .header { background: #fff8e1; border-bottom: 2px solid #ffc107; padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; }
        .header h1 { font-size: 18px; color: #e65100; }
        .header .tag { font-size: 12px; color: #795548; margin-left: 8px; }
        .review-switch { display: flex; align-items: center; gap: 8px; }
        .review-switch label { font-size: 12px; color: #666; }
        .review-switch .toggle { width: 44px; height: 24px; background: #ccc; border-radius: 12px; position: relative; cursor: pointer; }
        .review-switch .toggle.active { background: #ffc107; }
        .review-switch .toggle::after { content: ''; position: absolute; top: 2px; left: 2px; width: 20px; height: 20px; background: white; border-radius: 50%; transition: left 0.3s; }
        .review-switch .toggle.active::after { left: 22px; }
        .container { max-width: 800px; margin: 0 auto; padding: 24px; }
        .tab-bar { display: flex; border-bottom: 2px solid #e0e0e0; margin-bottom: 24px; background: white; border-radius: 8px 8px 0 0; }
        .tab-btn { padding: 12px 24px; border: none; background: none; font-size: 14px; color: #666; cursor: pointer; position: relative; }
        .tab-btn.active { color: #e65100; font-weight: bold; }
        .tab-btn.active::after { content: ''; position: absolute; bottom: -2px; left: 0; right: 0; height: 2px; background: #e65100; }
        .chat-container { background: white; border-radius: 0 0 8px 8px; min-height: 500px; display: flex; flex-direction: column; }
        .chat-messages { flex: 1; padding: 20px; overflow-y: auto; max-height: 400px; }
        .message { margin-bottom: 16px; display: flex; }
        .message.robot { justify-content: flex-start; }
        .message.user { justify-content: flex-end; }
        .message-content { max-width: 70%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; white-space: pre-line; }
        .message.robot .message-content { background: #f5f5f5; color: #333; border-bottom-left-radius: 4px; }
        .message.user .message-content { background: #ffc107; color: white; border-bottom-right-radius: 4px; }
        .chat-input-area { padding: 16px; background: #fafafa; border-top: 1px solid #e0e0e0; display: flex; gap: 12px; }
        .chat-input-area input { flex: 1; padding: 14px 18px; border: 1px solid #e0e0e0; border-radius: 25px; font-size: 14px; outline: none; }
        .chat-input-area input:focus { border-color: #ffc107; }
        .chat-input-area button { padding: 14px 24px; background: #ffc107; color: #fff; border: none; border-radius: 25px; font-size: 14px; cursor: pointer; font-weight: bold; }
        .chat-input-area button:hover { background: #ffb300; }
        .rounds-indicator { padding: 8px 16px; background: #fff8e1; text-align: center; font-size: 12px; color: #e65100; border-bottom: 1px solid #ffe082; }
        .review-list { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; }
        .review-item { padding: 16px; border-bottom: 1px solid #f0f0f0; }
        .review-item:last-child { border-bottom: none; }
        .review-item .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .review-item .intent { font-size: 14px; color: #e65100; font-weight: bold; }
        .review-item .confidence { font-size: 12px; color: #666; }
        .review-item .summary { font-size: 13px; color: #333; margin-bottom: 8px; }
        .review-item .meta { font-size: 11px; color: #9e9e9e; }
        .review-item .actions { display: flex; gap: 8px; margin-top: 12px; }
        .detail-panel { background: #fafafa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin-top: 8px; }
        .detail-panel .chat-history { font-size: 13px; color: #333; line-height: 1.8; max-height: 200px; overflow-y: auto; background: white; padding: 12px; border-radius: 4px; margin-top: 8px; }
        .detail-panel .chat-history .chat-line { margin-bottom: 4px; }
        .detail-panel .chat-history .chat-line.user { color: #e65100; }
        .detail-panel .chat-history .chat-line.assistant { color: #555; }
        .detail-panel .slots-info { font-size: 12px; color: #666; margin-top: 8px; }
        .detail-panel .slots-info .slot-tag { display: inline-block; background: #fff8e1; color: #e65100; padding: 2px 8px; border-radius: 4px; margin-right: 6px; }
        .select-intent { padding: 8px 12px; border: 1px solid #e0e0e0; border-radius: 4px; font-size: 13px; }
        .api-panel { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; }
        .api-panel .title { font-size: 16px; color: #e65100; padding: 16px; border-bottom: 1px solid #e0e0e0; }
        .api-group { padding: 16px; border-bottom: 1px solid #f0f0f0; }
        .api-group:last-child { border-bottom: none; }
        .api-group .group-name { font-size: 14px; color: #333; font-weight: bold; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px dashed #e0e0e0; }
        .api-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .api-table th, .api-table td { padding: 10px; text-align: left; border-bottom: 1px solid #f0f0f0; }
        .api-table th { background: #fafafa; color: #666; font-weight: normal; }
        .api-table .method { padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
        .api-table .method.post { background: #e3f2fd; color: #1565c0; }
        .api-table .method.get { background: #e8f5e9; color: #2e7d32; }
        .api-table .method.put { background: #fff8e1; color: #e65100; }
        .modal-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 1000; }
        .modal-content { background: white; border-radius: 12px; padding: 32px; max-width: 400px; width: 90%; text-align: center; }
        .modal-content h3 { font-size: 18px; color: #333; margin-bottom: 8px; }
        .modal-content p { font-size: 14px; color: #666; margin-bottom: 24px; white-space: pre-line; }
        .modal-content .btn { width: 100%; padding: 14px; border: none; border-radius: 6px; font-size: 14px; font-weight: bold; cursor: pointer; background: #ffc107; color: #fff; }
        .success-modal { background: #e8f5e9; border: 2px solid #4caf50; }
        .success-modal h3 { color: #2e7d32; }
        .success-modal .btn { background: #4caf50; }
        .btn-success { background: #4caf50; color: #fff; border: none; padding: 6px 12px; border-radius: 4px; font-size: 12px; cursor: pointer; }
        .btn-danger { background: #ef5350; color: #fff; border: none; padding: 6px 12px; border-radius: 4px; font-size: 12px; cursor: pointer; }
        .btn-secondary { background: #f0f0f0; color: #666; border: none; padding: 6px 12px; border-radius: 4px; font-size: 12px; cursor: pointer; }
        .btn-reset { background: #ff7043; color: #fff; border: none; padding: 8px 16px; border-radius: 20px; font-size: 12px; cursor: pointer; font-weight: bold; }
        .btn-reset:hover { background: #f4511e; }
        @media (max-width: 600px) {
            .header { flex-direction: column; gap: 12px; align-items: flex-start; }
            .message-content { max-width: 85%; }
            .api-table { font-size: 11px; }
            .api-table th, .api-table td { padding: 6px; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div style="display:flex; align-items:center;">
            <h1>客服中转中心</h1>
            <span class="tag">Service Router</span>
        </div>
        <div style="display:flex; align-items:center; gap:16px;">
            <button class="btn-reset" onclick="resetConversation()" title="开始新对话">重置对话</button>
            <div class="review-switch">
                <label>开启人工审核</label>
                <div class="toggle {{ 'active' if enable_human_review else '' }}" onclick="toggleReview()"></div>
            </div>
            <div class="review-switch">
                <label>长对话模式</label>
                <div class="toggle {{ 'active' if enable_long_conversation else '' }}" onclick="toggleLongConversation()"></div>
            </div>
        </div>
    </div>
    <div class="container">
        <div class="tab-bar">
            <form method="POST" style="display:inline;">
                <input type="hidden" name="view_switch" value="user">
                <button type="submit" class="tab-btn {{ 'active' if view == 'user' else '' }}">用户输入</button>
            </form>
            <form method="POST" style="display:inline;">
                <input type="hidden" name="view_switch" value="admin">
                <button type="submit" class="tab-btn {{ 'active' if view == 'admin' else '' }}">管理员审核</button>
            </form>
            <form method="POST" style="display:inline;">
                <input type="hidden" name="view_switch" value="api">
                <button type="submit" class="tab-btn {{ 'active' if view == 'api' else '' }}">API转发</button>
            </form>
        </div>

        {% if view == 'api' %}
            <div class="api-panel">
                <div class="title">API转发配置</div>
                {% for group in api_view.groups %}
                    <div class="api-group">
                        <div class="group-name">{{ group.value }}</div>
                        <table class="api-table">
                            <thead>
                                <tr>
                                    <th>渠道名称</th>
                                    <th>请求方式</th>
                                    <th>API地址</th>
                                    <th>描述</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for channel in api_view.channels %}
                                    {% if channel.group == group %}
                                        <tr>
                                            <td>{{ channel.name }}</td>
                                            <td><span class="method {{ channel.method.lower() }}">{{ channel.method }}</span></td>
                                            <td>{{ channel.api }}</td>
                                            <td>{{ channel.description }}</td>
                                        </tr>
                                    {% endif %}
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                {% endfor %}
            </div>

        {% elif view == 'admin' %}
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
                <h3>待审核列表</h3>
                <span style="font-size:12px; color:#9e9e9e;">共 {{ admin_view.total_pending }} 条待审核</span>
            </div>

            {% if admin_view.pending_reviews %}
                {% for review in admin_view.pending_reviews %}
                    <div class="review-item">
                        <div class="header">
                            <div class="intent">{{ review.intent }}</div>
                            <div class="confidence">置信度: {{ (review.confidence * 100) | round | int }}%</div>
                        </div>
                        <div class="summary">{{ review.conversation_summary }}</div>
                        <div class="meta">会话ID: {{ review.session_id[:8] }} | 对话轮次: {{ review.rounds }} | {{ review.created_at.strftime('%Y-%m-%d %H:%M:%S') }}</div>

                        <div class="detail-panel">
                            <strong style="font-size:13px; color:#333;">聊天记录：</strong>
                            <div class="chat-history">
                                {% for line in review.full_history.split('\n') %}
                                    {% if line.startswith('用户:') %}
                                        <div class="chat-line user">{{ line }}</div>
                                    {% elif line.startswith('客服:') %}
                                        <div class="chat-line assistant">{{ line }}</div>
                                    {% elif line %}
                                        <div class="chat-line">{{ line }}</div>
                                    {% endif %}
                                {% endfor %}
                            </div>
                            {% if review.slots %}
                                <div class="slots-info">
                                    <strong>已识别信息：</strong>
                                    {% for key, value in review.slots.items() %}
                                        <span class="slot-tag">{{ key }}: {{ value }}</span>
                                    {% endfor %}
                                </div>
                            {% endif %}
                        </div>

                        <div class="actions">
                            <form method="POST" style="display:inline;">
                                <input type="hidden" name="review_id" value="{{ review.id }}">
                                <select name="final_intent" class="select-intent">
                                    {% for intent in admin_view.intents %}
                                        <option value="{{ intent }}" {{ 'selected' if intent == review.intent else '' }}>{{ intent }}</option>
                                    {% endfor %}
                                </select>
                                <input type="hidden" name="admin_action" value="submit">
                                <button type="submit" class="btn-success">确认提交</button>
                            </form>
                        </div>
                    </div>
                {% endfor %}
            {% else %}
                <div style="text-align:center; padding:40px; color:#9e9e9e;">暂无待审核会话</div>
            {% endif %}

        {% else %}
            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    {% for msg in messages %}
                        {% if msg.role == 'user' %}
                            <div class="message user">
                                <div class="message-content">{{ msg.text }}</div>
                            </div>
                        {% elif msg.role == 'assistant' %}
                            <div class="message robot">
                                <div class="message-content">{{ msg.text }}</div>
                            </div>
                        {% endif %}
                    {% endfor %}
                </div>
                <div class="rounds-indicator">
                    对话进行中，还需要 {{ remaining_rounds }} 轮对话后总结判断
                </div>
                <div class="chat-input-area">
                    <form method="POST" style="width:100%; display:flex; gap:12px;" id="chatForm" onsubmit="return handleSubmit(this)">
                        <input type="text" name="user_text" id="userInput" placeholder="请输入您的需求..." autocomplete="off" onkeydown="if(event.keyCode==13) this.form.submit()">
                        <button type="submit" id="sendBtn" onclick="var b=this; setTimeout(function(){b.disabled=true; b.textContent='发送中...';}, 0);">发送</button>
                    </form>
                </div>
            </div>
        {% endif %}
    </div>

    {% if show_redirect_modal %}
        <div class="modal-overlay" id="redirectModal">
            <div class="modal-content">
                <h3>{{ redirect_channel }}</h3>
                <p>正在为您跳转到对应服务渠道...</p>
                <div style="margin-top:16px; color:#9e9e9e; font-size:12px;">目标API: {{ redirect_api }}</div>
                <button class="btn" onclick="closeModal()">关闭</button>
            </div>
        </div>
    {% endif %}

    {% if show_success_modal %}
        <div class="modal-overlay" id="successModal">
            <div class="modal-content success-modal">
                <h3>分发成功</h3>
                <p>{{ success_message }}</p>
                <button class="btn" onclick="closeSuccessModal()">关闭</button>
            </div>
        </div>
    {% endif %}

    <script>
        function handleSubmit(form) {
            var input = document.getElementById('userInput');
            if (!input || !input.value.trim()) {
                return false;
            }
            var btn = document.getElementById('sendBtn');
            if (btn) {
                btn.disabled = true;
                btn.textContent = '发送中...';
            }
            return true;
        }

        function resetConversation() {
            if (confirm('确定要开始新对话吗？当前对话记录将被清除。')) {
                fetch('/reset', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'}
                }).then(res => res.json()).then(data => {
                    if (data.success) {
                        window.location.reload();
                    }
                });
            }
        }

        function toggleReview() {
            fetch('/', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: 'review_switch=toggle'
            }).then(res => res.json()).then(data => {
                var toggle = document.querySelector('.review-switch:nth-child(2) .toggle');
                if (data.enable_human_review) {
                    toggle.classList.add('active');
                } else {
                    toggle.classList.remove('active');
                }
            });
        }

        function toggleLongConversation() {
            fetch('/', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: 'long_conversation_switch=toggle'
            }).then(res => res.json()).then(data => {
                var toggle = document.querySelector('.review-switch:nth-child(3) .toggle');
                if (data.enable_long_conversation) {
                    toggle.classList.add('active');
                } else {
                    toggle.classList.remove('active');
                }
            });
        }

        function closeModal() {
            var modal = document.getElementById('redirectModal');
            if (modal) {
                modal.style.display = 'none';
            }
        }

        function closeSuccessModal() {
            var modal = document.getElementById('successModal');
            if (modal) {
                modal.style.display = 'none';
            }
        }

        window.onload = function() {
            var chatMessages = document.getElementById('chatMessages');
            if (chatMessages) {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            var userInput = document.getElementById('userInput');
            if (userInput) {
                userInput.focus();
            }
        };
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