from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
from models.database import db, init_db
from models.video_analysis import VideoAnalysis
from routes.upload_routes import upload_bp
from routes.heatmap_routes import heatmap_bp
import logging

# 앱 초기화
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev_key')
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(os.getcwd(), 'processed')

# 데이터베이스 설정 - MariaDB 연결
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mysql+pymysql://{os.getenv('DB_USER', 'admin')}:{os.getenv('DB_PASS', 'admin')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME', 'heatmap_db')}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 필요한 디렉토리 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['PROCESSED_FOLDER'], 'frames'), exist_ok=True)
os.makedirs(os.path.join(app.config['PROCESSED_FOLDER'], 'heatmaps'), exist_ok=True)
os.makedirs(os.path.join(app.config['PROCESSED_FOLDER'], 'videos'), exist_ok=True)

# 데이터베이스 초기화
init_db(app)

# 템플릿 컨텍스트 프로세서 추가
@app.context_processor
def utility_processor():
    def now():
        from datetime import datetime
        return datetime.now()
    return {'now': now}

# 블루프린트 등록
app.register_blueprint(upload_bp)
app.register_blueprint(heatmap_bp)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return redirect(url_for('upload_bp.upload_page'))

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5500))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', 'True').lower() == 'true') 
