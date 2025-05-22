from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
import os
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import threading
from models.database import db
from models.video_analysis import VideoAnalysis
from models.yolo_detector import YOLODetector
import logging

# 블루프린트 설정
upload_bp = Blueprint('upload_bp', __name__, url_prefix='/upload')
logger = logging.getLogger(__name__)

# 상수 정의
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'mkv'}
UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB
PROGRESS_LOG_INTERVAL = 10  # 10% 단위로 진행률 로깅
RECENT_UPLOADS_LIMIT = int(os.getenv('RECENT_UPLOADS_LIMIT', '10'))

def allowed_file(filename):
    """파일 확장자 검사"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@upload_bp.route('/', methods=['GET'])
def upload_page():
    """업로드 페이지 표시"""
    # 최근 업로드된 비디오 목록
    recent_uploads = VideoAnalysis.query.order_by(VideoAnalysis.upload_date.desc()).limit(RECENT_UPLOADS_LIMIT).all()
    return render_template('upload/index.html', recent_uploads=recent_uploads)

@upload_bp.route('/process', methods=['POST'])
def process_video():
    """비디오 업로드 및 처리"""
    try:
        if 'video' not in request.files:
            flash('비디오 파일이 없습니다.', 'danger')
            return redirect(request.url)
        
        file = request.files['video']
        
        if file.filename == '':
            flash('파일이 선택되지 않았습니다.', 'danger')
            return redirect(request.url)
        
        if not allowed_file(file.filename):
            flash(f'허용되지 않는 파일 형식입니다. 허용된 형식: {", ".join(ALLOWED_EXTENSIONS)}', 'danger')
            return redirect(request.url)
        
        # 고유 ID 생성
        upload_id = str(uuid.uuid4())
        
        # 파일명 보안 처리 및 저장
        original_filename = secure_filename(file.filename)
        
        # 파일명에서 확장자 분리
        name, ext = os.path.splitext(original_filename)
        
        # 날짜시간 형식 설정 -> 20250521_103045
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 새 파일명 생성: 원본이름_날짜시간_ID.확장자
        stored_filename = f"{name}_{timestamp}_{upload_id[:8]}{ext}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], stored_filename)
        
        logger.info(f"업로드 처리: 원본={original_filename}, 저장파일명={stored_filename}")
        
        # 파일을 대용량 청크 단위로 저장
        try:
            # 파일 크기 확인
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)  # 파일 포인터 초기화
            
            logger.info(f"업로드 시작: {original_filename}, 크기: {file_size/1024/1024:.1f}MB")
            with open(file_path, 'wb') as f:
                bytes_written = 0
                while True:
                    chunk = file.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_written += len(chunk)
                    # 진행률 로깅
                    progress = (bytes_written / file_size) * 100
                    if progress % PROGRESS_LOG_INTERVAL < (UPLOAD_CHUNK_SIZE / file_size) * 100:
                        logger.info(f"업로드 진행률: {int(progress)}%")
                        
            logger.info(f"업로드 완료: {original_filename}, 저장경로: {file_path}")
        except Exception as e:
            logger.error(f"파일 저장 중 오류: {e}")
            flash(f'파일 저장 중 오류가 발생했습니다: {str(e)}', 'danger')
            # 실패한 경우 부분적으로 저장된 파일 삭제
            if os.path.exists(file_path):
                os.remove(file_path)
            return redirect(request.url)
        
        # 데이터베이스에 분석 정보 저장
        analysis = VideoAnalysis(
            upload_id=upload_id,
            original_filename=original_filename,
            stored_filename=stored_filename
        )
        
        # 추가 메타데이터 저장
        analysis.file_size = file_size
        analysis.upload_date = datetime.now()
        analysis.status = 'pending'
        
        db.session.add(analysis)
        db.session.commit()
        
        logger.info(f"데이터베이스 저장 완료: ID={upload_id}, 파일={original_filename}")
        
        # 백그라운드에서 비디오 처리
        detector = YOLODetector()
        thread = threading.Thread(target=detector.process_video, 
                                args=(file_path, upload_id, current_app._get_current_object()))
        thread.daemon = True
        thread.start()
        
        flash('비디오가 업로드되었습니다. 처리 중입니다...', 'success')
        return redirect(url_for('heatmap_bp.view_analysis', upload_id=upload_id))
        
    except Exception as e:
        logger.error(f"비디오 업로드 중 오류 발생: {e}")
        flash(f'비디오 업로드 중 오류 발생: {str(e)}', 'danger')
        return redirect(request.url)

@upload_bp.route('/status/<upload_id>', methods=['GET'])
def check_status(upload_id):
    """비디오 처리 상태 확인"""
    analysis = VideoAnalysis.query.filter_by(upload_id=upload_id).first()
    
    if not analysis:
        return jsonify({'status': 'error', 'message': '업로드 정보를 찾을 수 없습니다.'}), 404
    
    return jsonify({
        'status': analysis.status,
        'total_persons': analysis.total_persons,
        'error': analysis.error_message
    }) 