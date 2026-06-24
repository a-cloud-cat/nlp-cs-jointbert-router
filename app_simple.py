from flask import Flask, request, render_template_string, session as flask_session, jsonify
import uuid

app = Flask(__name__)
app.secret_key = 'test'

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'session_id' not in flask_session:
        flask_session['session_id'] = str(uuid.uuid4())
    
    messages = flask_session.get('messages', [])
    user_message_count = len([m for m in messages if m['role'] == 'user'])
    
    if request.method == 'POST':
        user_text = request.form.get('user_text', '').strip()
        if user_text:
            messages.append({'role': 'user', 'text': user_text})
            
            if user_message_count >= 4:
                response = '根据您的描述，我已为您跳转至对应服务渠道'
            else:
                response = '请问您能详细说说吗？'
            
            messages.append({'role': 'assistant', 'text': response})
            flask_session['messages'] = messages
    
    messages = flask_session.get('messages', [])
    user_message_count = len([m for m in messages if m['role'] == 'user'])
    
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <style>
        .chat-container { max-width: 600px; margin: 0 auto; }
        .message { margin: 10px; padding: 10px; border-radius: 8px; }
        .user { background: #ffc107; color: white; text-align: right; }
        .robot { background: #f5f5f5; color: #333; }
        .rounds { text-align: center; color: #666; font-size: 12px; }
        input { width: 100%; padding: 10px; margin: 10px 0; }
        button { padding: 10px 20px; background: #ffc107; color: white; border: none; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="rounds">还需要 {{ max(0, 5 - user_message_count) }} 轮对话后总结判断</div>
        {% for msg in messages %}
            <div class="message {{ 'user' if msg.role == 'user' else 'robot' }}">
                {{ msg.text }}
            </div>
        {% endfor %}
        <form method="POST">
            <input type="text" name="user_text" placeholder="请输入...">
            <button type="submit">发送</button>
        </form>
    </div>
</body>
</html>
""", messages=messages, user_message_count=user_message_count)

if __name__ == '__main__':
    app.run(port=5001, debug=True)