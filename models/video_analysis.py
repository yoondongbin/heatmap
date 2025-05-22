from models.database import db
from datetime import datetime
import json

class VideoAnalysis(db.Model):
    """비디오 분석 결과 모델"""
    id = db.Column(db.Integer, primary_key=True)
    upload_id = db.Column(db.String(50), nullable=False, unique=True)  # 업로드 식별자
    original_filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.BigInteger)  # 파일 크기
    processed_video_path = db.Column(db.String(255))  # 처리된 기본 동영상 경로
    heatmap_video_path = db.Column(db.String(255))  # 히트맵이 표시된 동영상 경로
    bounding_box_video_path = db.Column(db.String(255))  # 바운딩 박스가 표시된 동영상 경로
    
    upload_date = db.Column(db.DateTime, default=datetime.now)
    analysis_date = db.Column(db.DateTime)
    duration = db.Column(db.Float)
    
    # 태그 및 설명
    tags = db.Column(db.String(255)) 
    description = db.Column(db.Text)
    
    # 분석 결과 요약
    total_persons = db.Column(db.Integer, default=0)  # 총 탐지된 사람 수
    max_persons = db.Column(db.Integer, default=0)  # 최대 동시 탐지 사람 수
    frame_count = db.Column(db.Integer, default=0)  # 총 프레임 수
    heatmap_path = db.Column(db.String(255))  # 전체 히트맵 이미지 경로
    
    # 프레임별 히트맵 및 분석 결과 JSON -> path: 경로, frame_number: 프레임 번호, persons: 탐지된 사람 수
    frame_data = db.Column(db.Text(length=None))  # MySQL LONGTEXT 타입으로 매핑됨
    
    # 시간대별 통계 -> 10초 간격으로 탐지된 사람 수
    time_stats = db.Column(db.Text(length=None))  # MySQL LONGTEXT 타입으로 매핑됨
    
    # 이동 경로 데이터 -> 그리드 기반 이동 통계
    movement_data = db.Column(db.Text(length=None))  # MySQL LONGTEXT 타입으로 매핑됨
    
    # 분석 상태 -> pending, processing, completed, failed
    status = db.Column(db.String(20), default='pending')
    error_message = db.Column(db.Text)
    
    def __init__(self, upload_id, original_filename, stored_filename):
        self.upload_id = upload_id
        self.original_filename = original_filename
        self.stored_filename = stored_filename
        
    def set_frame_data(self, frame_data):
        """프레임 데이터 저장"""
        self.frame_data = json.dumps(frame_data)
        
    def get_frame_data(self):
        """프레임 데이터 조회"""
        if self.frame_data:
            return json.loads(self.frame_data)
        return []
        
    def set_time_stats(self, time_stats):
        """시간대별 통계 저장"""
        self.time_stats = json.dumps(time_stats)
        
    def get_time_stats(self):
        """시간대별 통계 조회"""
        if self.time_stats:
            return json.loads(self.time_stats)
        return {}
        
    def set_movement_data(self, movement_data):
        """이동 경로 데이터 저장"""
        self.movement_data = json.dumps(movement_data)
        
    def get_movement_data(self):
        """이동 경로 데이터 조회"""
        if self.movement_data:
            return json.loads(self.movement_data)
        return {}
        
    def to_dict(self):
        """모델을 딕셔너리로 변환"""
        return {
            'id': self.id,
            'upload_id': self.upload_id,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'file_size_mb': round(self.file_size / 1024 / 1024, 2) if self.file_size else None,
            'upload_date': self.upload_date.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_date': self.analysis_date.strftime('%Y-%m-%d %H:%M:%S') if self.analysis_date else None,
            'duration': self.duration,
            'total_persons': self.total_persons,
            'max_persons': self.max_persons,
            'frame_count': self.frame_count,
            'status': self.status,
            'tags': self.tags,
            'description': self.description,
            'heatmap_path': self.heatmap_path,
            'processed_video_path': self.processed_video_path,
            'heatmap_video_path': self.heatmap_video_path,
            'bounding_box_video_path': self.bounding_box_video_path
        } 