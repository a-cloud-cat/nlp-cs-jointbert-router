import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app.app import app

if __name__ == '__main__':
    print('=' * 50)
    print('客服中转中心已启动！')
    print('访问地址: http://127.0.0.1:5000')
    print('=' * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)