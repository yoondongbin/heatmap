from flask import Blueprint, render_template, request, jsonify, redirect, url_for, send_from_directory, current_app, flash, send_file, Response
import os
from models.database import db
from models.video_analysis import VideoAnalysis
import logging
import shutil
import re

# 블루프린트 설정
heatmap_bp = Blueprint('heatmap_bp', __name__, url_prefix='/heatmap')
logger = logging.getLogger(__name__)

@heatmap_bp.route('/', methods=['GET'])
def heatmap_list():
    """분석 완료된 히트맵 목록 페이지"""
    # 완료된 분석 목록
    completed_analyses = VideoAnalysis.query.filter_by(status='completed').order_by(VideoAnalysis.analysis_date.desc()).all()
    return render_template('heatmap/list.html', analyses=completed_analyses)

@heatmap_bp.route('/view/<upload_id>', methods=['GET'])
def view_analysis(upload_id):
    """분석 결과 보기"""
    analysis = VideoAnalysis.query.filter_by(upload_id=upload_id).first_or_404()
    return render_template('heatmap/view.html', analysis=analysis)

@heatmap_bp.route('/delete/<upload_id>', methods=['POST'])
def delete_analysis(upload_id):
    """분석 결과 및 관련 파일 삭제"""
    analysis = VideoAnalysis.query.filter_by(upload_id=upload_id).first_or_404()
    
    try:
        # 관련 파일 삭제
        files_to_delete = []
        
        # 원본 비디오 파일
        if hasattr(analysis, 'stored_filename') and analysis.stored_filename:
            original_video = os.path.join(current_app.config['UPLOAD_FOLDER'], analysis.stored_filename)
            files_to_delete.append(original_video)
        
        # 처리된 비디오 파일들
        for attr in ['processed_video_path', 'heatmap_video_path', 'bounding_box_video_path']:
            if hasattr(analysis, attr) and getattr(analysis, attr):
                files_to_delete.append(getattr(analysis, attr))
        
        # 히트맵 이미지
        if hasattr(analysis, 'heatmap_path') and analysis.heatmap_path:
            files_to_delete.append(analysis.heatmap_path)
        
        # 프레임 이미지, 히트맵 이미지 디렉토리
        frame_dir = os.path.join(current_app.config['PROCESSED_FOLDER'], 'frames')
        heatmap_dir = os.path.join(current_app.config['PROCESSED_FOLDER'], 'heatmaps')
        
        # 업로드 ID로 시작하는 프레임 이미지 파일들 삭제
        if os.path.exists(frame_dir):
            for filename in os.listdir(frame_dir):
                if filename.startswith(f"{upload_id}_"):
                    files_to_delete.append(os.path.join(frame_dir, filename))
        
        # 업로드 ID로 시작하는 히트맵 이미지 파일들 삭제
        if os.path.exists(heatmap_dir):
            for filename in os.listdir(heatmap_dir):
                if filename.startswith(f"{upload_id}_"):
                    files_to_delete.append(os.path.join(heatmap_dir, filename))
        
        # 파일 삭제 실행
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                logger.info(f"파일 삭제됨: {file_path}")
        
        # 데이터베이스에서 삭제
        db.session.delete(analysis)
        db.session.commit()
        
        # 성공 메시지
        flash('분석 결과가 성공적으로 삭제되었습니다.', 'success')
        return redirect(url_for('heatmap_bp.heatmap_list'))
    
    except Exception as e:
        logger.error(f"삭제 중 오류 발생: {e}")
        db.session.rollback()
        flash(f'삭제 중 오류가 발생했습니다: {str(e)}', 'danger')
        return redirect(url_for('heatmap_bp.heatmap_list'))

@heatmap_bp.route('/data/<upload_id>', methods=['GET'])
def get_analysis_data(upload_id):
    """분석 데이터 API"""
    analysis = VideoAnalysis.query.filter_by(upload_id=upload_id).first_or_404()
    
    frame_data = analysis.get_frame_data()
    time_stats = analysis.get_time_stats()
    movement_data = analysis.get_movement_data()
    
    # 프레임 데이터에서 인원수가 없는 경우 0으로 설정
    if frame_data:
        for frame in frame_data:
            if 'persons' not in frame or frame['persons'] is None:
                frame['persons'] = 0
    
    # 시간 통계에서 평균 인원수가 없는 경우 0으로 설정
    if time_stats:
        for time_key in time_stats:
            if 'avg_persons' not in time_stats[time_key] or time_stats[time_key]['avg_persons'] is None:
                time_stats[time_key]['avg_persons'] = 0
    
    return jsonify({
        'upload_id': analysis.upload_id,
        'status': analysis.status,
        'original_filename': analysis.original_filename,
        'upload_date': analysis.upload_date.strftime('%Y-%m-%d %H:%M:%S'),
        'analysis_date': analysis.analysis_date.strftime('%Y-%m-%d %H:%M:%S') if analysis.analysis_date else None,
        'duration': analysis.duration,
        'total_persons': analysis.total_persons or 0,
        'max_persons': analysis.max_persons or 0,
        'processed_video_path': analysis.processed_video_path,
        'heatmap_video_path': analysis.heatmap_video_path,
        'bounding_box_video_path': analysis.bounding_box_video_path,
        'heatmap_path': analysis.heatmap_path,
        'frame_data': frame_data,
        'time_stats': time_stats,
        'movement_data': movement_data
    })

@heatmap_bp.route('/video/<type>/<upload_id>', methods=['GET'])
def get_video(type, upload_id):
    """비디오 파일 직접 제공 - 단순화된 방식"""
    
    analysis = VideoAnalysis.query.filter_by(upload_id=upload_id).first_or_404()
    
    # 요청된 비디오 유형에 따른 파일 경로 결정
    video_path = None
    if type == 'processed':
        video_path = analysis.processed_video_path
    elif type == 'heatmap':
        video_path = analysis.heatmap_video_path
    elif type == 'bounding_box':
        video_path = analysis.bounding_box_video_path
    else:
        return "잘못된 비디오 유형입니다.", 400
    
    # 파일 존재 여부 확인
    if not video_path or not os.path.exists(video_path):
        logger.error(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        return "비디오 파일을 찾을 수 없습니다.", 404
    
    video_dir = os.path.dirname(video_path)
    video_filename = os.path.basename(video_path)
    
    # 다운로드 옵션 확인
    download_mode = request.args.get('download', '0')
    as_attachment = download_mode == '1'
    
    # 로그 기록
    logger.info(f"비디오 요청: {type}/{upload_id}, 다운로드 모드: {as_attachment}")
    
    # 가장 단순한 방식으로 파일 제공
    return send_from_directory(
        video_dir, 
        video_filename,
        as_attachment=as_attachment,
        download_name=f"{analysis.original_filename}_{type}.mp4" if as_attachment else None,
        mimetype='video/mp4',
        conditional=True  # Range 요청 지원
    )

@heatmap_bp.route('/processed-video/<upload_id>', methods=['GET'])
def get_processed_video(upload_id):
    """처리된 비디오 스트리밍 (이전 버전과의 호환성)"""
    return get_video('processed', upload_id)

@heatmap_bp.route('/image/<type>/<upload_id>/<frame_number>', methods=['GET'])
def get_frame_image(type, upload_id, frame_number):
    """프레임 또는 히트맵 이미지 가져오기"""
    if type not in ['frames', 'heatmaps']:
        return "잘못된 이미지 유형입니다.", 400
    
    frame_number = int(frame_number)
    
    # 분석 데이터 조회
    analysis = VideoAnalysis.query.filter_by(upload_id=upload_id).first_or_404()
    frame_data = analysis.get_frame_data()
    
    # 분석 타임스탬프 추출 (가장 최근 분석 데이터)
    timestamp = None
    if frame_data and 'analysis_timestamp' in frame_data[0]:
        timestamp = frame_data[0]['analysis_timestamp']
    
    # 타임스탬프가 있으면 파일명에 포함
    image_filename = None
    if timestamp:
        image_filename = f"{upload_id}_frame_{frame_number}_{timestamp}.jpg"
    else:
        # 이전 버전 호환성 유지
        image_filename = f"{upload_id}_frame_{frame_number}.jpg"
    
    image_path = os.path.join(current_app.config['PROCESSED_FOLDER'], type, image_filename)
    
    if not os.path.exists(image_path):
        logger.error(f"이미지를 찾을 수 없습니다: {image_path}")
        return "이미지를 찾을 수 없습니다.", 404
    
    return send_from_directory(os.path.join(current_app.config['PROCESSED_FOLDER'], type), image_filename)

@heatmap_bp.route('/final-heatmap/<upload_id>', methods=['GET'])
def get_final_heatmap(upload_id):
    """최종 히트맵 이미지 가져오기"""
    # 분석 데이터 조회
    analysis = VideoAnalysis.query.filter_by(upload_id=upload_id).first_or_404()
    
    # 히트맵 파일 경로가 DB에 저장되어 있음
    if not analysis.heatmap_path or not os.path.exists(analysis.heatmap_path):
        logger.error(f"히트맵 이미지를 찾을 수 없습니다: {analysis.heatmap_path}")
        return "히트맵 이미지를 찾을 수 없습니다.", 404
    
    image_dir = os.path.dirname(analysis.heatmap_path)
    image_filename = os.path.basename(analysis.heatmap_path)
    
    return send_from_directory(image_dir, image_filename) 