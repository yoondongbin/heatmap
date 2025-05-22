import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
from models.database import db
from models.video_analysis import VideoAnalysis

# 상수 정의
TARGET_FPS = float(os.getenv('TARGET_FPS', '6.0'))  # 1초당 처리할 프레임 수
HEATMAP_FRAME_INTERVAL = 10  # 히트맵 추출 간격
TARGET_HEIGHT = 720  # 720p 해상도
TARGET_WIDTH = 1280  # 16:9 비율
GRID_SIZE = int(os.getenv('GRID_SIZE', '20'))  # 이동 경로 그리드 크기
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.3'))  # YOLO 신뢰도 임계값

class YOLODetector:
    """YOLOv8 모델을 사용한 객체 감지 및 히트맵 생성 클래스"""
    
    def __init__(self, model_path='yolov8m.pt'):
        """
        YOLO 모델 초기화
        Args:
            model_path: YOLO 모델 경로 (기본값: yolov8m.pt)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("YOLOv8 모델 로드 중...")
        
        try:
            self.model = YOLO(model_path)
            self.logger.info("YOLOv8 모델 로드 완료")
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {e}")
            raise
    
    def process_video(self, video_path, upload_id, app):
        """
        비디오 파일 처리 및 히트맵 생성
        Args:
            video_path: 업로드된 비디오 파일 경로
            upload_id: 업로드 식별자
            app: Flask 앱 컨텍스트
        """
        with app.app_context():
            try:
                analysis = VideoAnalysis.query.filter_by(upload_id=upload_id).first()
                if not analysis:
                    self.logger.error(f"업로드 ID {upload_id}에 해당하는 분석 데이터를 찾을 수 없습니다.")
                    return
                
                analysis.status = 'processing'
                db.session.commit()
                
                # 비디오 캡처 객체 초기화
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    raise Exception("비디오 파일을 열 수 없습니다.")
                
                # 비디오 속성 가져오기
                original_fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / original_fps
                
                # 설정 - 1초당 6프레임 처리
                target_fps = TARGET_FPS
                process_every_n_frames = max(1, int(original_fps / target_fps))

                # 10프레임 단위로 히트맵 추출
                heatmap_frame_interval = HEATMAP_FRAME_INTERVAL

                # 720p로 리사이징
                target_height = TARGET_HEIGHT
                target_width = TARGET_WIDTH
                resized_size = (target_width, target_height)
                
                # 추가 정보
                self.logger.info(f"총 프레임 수: {frame_count}, 원본 FPS: {original_fps}, 추출 간격: {process_every_n_frames}프레임, 히트맵 간격: {heatmap_frame_interval}프레임")
                
                # 비디오 작성자 초기화
                # XVID 코덱은 호환성 문제가 있어 FFMPEG를 통해 H.264 인코딩으로 변경
                self.logger.info("비디오 임시 파일 설정...")
                temp_processed_video = os.path.join(app.config['PROCESSED_FOLDER'], 'videos', f"{upload_id}_processed_temp.avi")
                temp_heatmap_video = os.path.join(app.config['PROCESSED_FOLDER'], 'videos', f"{upload_id}_heatmap_temp.avi")
                temp_bbox_video = os.path.join(app.config['PROCESSED_FOLDER'], 'videos', f"{upload_id}_bbox_temp.avi")

                # 타임스탬프를 파일명에 추가하여 분석 버전 구분
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                processed_video_path = os.path.join(app.config['PROCESSED_FOLDER'], 'videos', f"{upload_id}_processed_{timestamp}.mp4")
                heatmap_video_path = os.path.join(app.config['PROCESSED_FOLDER'], 'videos', f"{upload_id}_heatmap_{timestamp}.mp4")
                bounding_box_video_path = os.path.join(app.config['PROCESSED_FOLDER'], 'videos', f"{upload_id}_bbox_{timestamp}.mp4")

                # 임시 파일용 비디오 작성자 -> MJPG는 대부분의 시스템에서 지원
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out_processed = cv2.VideoWriter(temp_processed_video, fourcc, target_fps, resized_size)
                out_heatmap = cv2.VideoWriter(temp_heatmap_video, fourcc, target_fps, resized_size)
                out_bbox = cv2.VideoWriter(temp_bbox_video, fourcc, target_fps, resized_size)
                
                # 히트맵 초기화
                heatmap = np.zeros((target_height, target_width), dtype=np.float32)
                current_heatmap = np.zeros((target_height, target_width), dtype=np.float32)
                
                # 프레임 데이터 저장 리스트
                frame_data_list = []
                
                # 시간대별 통계
                time_stats = {}
                
                # 이동 경로 통계 -> 좌표 그리드 기반
                grid_size = GRID_SIZE
                grid_width = target_width // grid_size
                grid_height = target_height // grid_size
                movement_stats = np.zeros((grid_size, grid_size), dtype=int)
                
                # 배경 이미지 -> 첫 프레임
                ret, first_frame = cap.read()
                if ret:
                    background = cv2.resize(first_frame, resized_size)
                    background_path = os.path.join(app.config['PROCESSED_FOLDER'], 'frames', f"{upload_id}_background.jpg")
                    cv2.imwrite(background_path, background)
                else:
                    raise Exception("비디오에서 프레임을 읽을 수 없습니다.")
                
                # 비디오를 처음부터 다시 시작
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # 총 탐지된 사람 수
                total_persons = 0
                max_persons_in_frame = 0
                
                # 프레임 처리
                frame_idx = 0
                processed_frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # N프레임마다 처리
                    if frame_idx % process_every_n_frames == 0:
                        # 프레임 리사이징 -> 720p로
                        resized_frame = cv2.resize(frame, resized_size)
                        
                        # 각 프레임마다 현재 히트맵 초기화
                        current_heatmap = np.zeros_like(current_heatmap)
                        
                        # YOLO 탐지 실행
                        results = self.model.predict(resized_frame, conf=CONFIDENCE_THRESHOLD, classes=0)  # 클래스 0 = 사람
                        
                        # 결과 처리
                        persons_in_frame = 0
                        detections = []
                        
                        # 바운딩 박스가 있는 프레임 복사
                        bbox_frame = resized_frame.copy()
                        
                        for result in results:
                            boxes = result.boxes.cpu().numpy()
                            
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                                confidence = box.conf[0]
                                cls = int(box.cls[0])
                                
                                # 사람 객체만 처리 -> 클래스 0
                                if cls == 0:  
                                    persons_in_frame += 1
                                    
                                    # 히트맵에 탐지 위치 추가 -> 사람의 아래쪽 중앙 부분
                                    center_x = (x1 + x2) // 2
                                    foot_y = y2
                                    
                                    # 좌표가 이미지 범위 내에 있는지 확인
                                    center_x = min(max(0, center_x), target_width-1)
                                    foot_y = min(max(0, foot_y), target_height-1)
                                    
                                    # 이동 경로 통계 업데이트
                                    grid_x = min(center_x // grid_width, grid_size-1)
                                    grid_y = min(foot_y // grid_height, grid_size-1)
                                    movement_stats[grid_y, grid_x] += 1
                                    
                                    # 히트맵에 가중치 추가 -> 가우시안 블러 적용
                                    sigma = max(int((y2 - y1) / 4), 5)  # 바운딩 박스 높이에 비례하는 시그마
                                    temp_heatmap = np.zeros_like(heatmap)
                                    temp_heatmap[foot_y, center_x] = 1
                                    temp_heatmap = cv2.GaussianBlur(temp_heatmap, (0, 0), sigma)
                                    
                                    # 전체 히트맵과 현재 프레임 히트맵 업데이트
                                    heatmap += temp_heatmap
                                    current_heatmap += temp_heatmap
                                    
                                    # 바운딩 박스 그리기 -> 바운딩 박스 비디오용
                                    cv2.rectangle(bbox_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(bbox_frame, f"{confidence:.2f}", (x1, y1 - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    
                                    # 탐지 정보 저장
                                    detections.append({
                                        'x1': int(x1), 'y1': int(y1), 
                                        'x2': int(x2), 'y2': int(y2),
                                        'confidence': float(confidence)
                                    })
                        
                        # 최대 동시 탐지 사람 수 업데이트
                        max_persons_in_frame = max(max_persons_in_frame, persons_in_frame)
                        total_persons += persons_in_frame
                        
                        # 시간 통계 업데이트 -> 1초 단위
                        time_in_seconds = frame_idx / original_fps
                        time_bucket = int(time_in_seconds)  # 1초 단위로 설정
                        time_key = f"{time_bucket}s"
                        
                        if time_key not in time_stats:
                            time_stats[time_key] = {'persons': 0, 'frames': 0}
                        
                        time_stats[time_key]['persons'] += persons_in_frame
                        time_stats[time_key]['frames'] += 1
                        
                        # 10프레임마다 히트맵 이미지 저장
                        if processed_frame_count % heatmap_frame_interval == 0:
                            # 타임스탬프를 포함하여 파일명 유일성 보장 -> 파일명 유일성 보장
                            frame_heatmap_path = os.path.join(app.config['PROCESSED_FOLDER'], 'heatmaps', f"{upload_id}_frame_{frame_idx}_{timestamp}.jpg")
                            heatmap_frame = self._create_heatmap_overlay(current_heatmap, resized_frame.copy())
                            cv2.imwrite(frame_heatmap_path, heatmap_frame)
                            
                            # 프레임 이미지 저장
                            frame_path = os.path.join(app.config['PROCESSED_FOLDER'], 'frames', f"{upload_id}_frame_{frame_idx}_{timestamp}.jpg")
                            cv2.imwrite(frame_path, resized_frame)
                            
                            # 프레임 데이터 저장
                            frame_data = {
                                'frame_number': frame_idx,
                                'timestamp': frame_idx / original_fps,
                                'persons': persons_in_frame,
                                'frame_path': frame_path,
                                'heatmap_path': frame_heatmap_path,
                                'detections': detections,
                                'analysis_timestamp': timestamp  # 분석 타임스탬프 추가 -> 분석 타임스탬프 추가
                            }
                            frame_data_list.append(frame_data)
                        
                        # 로그 출력
                        if frame_idx % 100 == 0:
                            self.logger.info(f"프레임 {frame_idx}/{frame_count} 처리 중... (진행률: {frame_idx/frame_count*100:.1f}%)")
                        
                        # 3가지 비디오에 프레임 저장 -> 3가지 비디오에 프레임 저장
                        out_processed.write(resized_frame)
                        out_heatmap.write(self._create_heatmap_overlay(current_heatmap, resized_frame.copy()))
                        out_bbox.write(bbox_frame)
                    
                    processed_frame_count += 1
                    frame_idx += 1
                
                # 리소스 해제
                cap.release()
                out_processed.release()
                out_heatmap.release()
                out_bbox.release()
                
                # 임시 비디오 파일을 H.264 MP4로 변환
                try:
                    self.logger.info("임시 비디오 파일을 H.264 MP4로 변환 중...")
                    
                    # FFmpeg 명령어 실행 함수
                    def convert_video(input_path, output_path, label=""):
                        self.logger.info(f"{label} 비디오 변환 중: {input_path} -> {output_path}")
                        import subprocess
                        cmd = [
                            'ffmpeg', '-i', input_path, 
                            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                            '-y', output_path
                        ]
                        subprocess.run(cmd, check=True)
                        # 변환 후 임시 파일 삭제
                        if os.path.exists(input_path):
                            os.remove(input_path)
                    
                    # 각 비디오 변환
                    convert_video(temp_processed_video, processed_video_path, "기본 처리")
                    convert_video(temp_heatmap_video, heatmap_video_path, "히트맵")
                    convert_video(temp_bbox_video, bounding_box_video_path, "바운딩 박스")
                    
                except Exception as e:
                    self.logger.error(f"비디오 변환 중 오류 발생: {e}")
                    # 오류 발생 시 임시 파일 경로를 최종 경로로 사용
                    processed_video_path = temp_processed_video
                    heatmap_video_path = temp_heatmap_video
                    bounding_box_video_path = temp_bbox_video
                
                # 평균 시간대별 통계 계산
                for time_key in time_stats:
                    if time_stats[time_key]['frames'] > 0:
                        time_stats[time_key]['avg_persons'] = time_stats[time_key]['persons'] / time_stats[time_key]['frames']
                
                # 이동 경로 데이터 정규화 및 저장
                movement_path_data = {}
                if np.max(movement_stats) > 0:
                    normalized_movement = movement_stats / np.max(movement_stats)
                    for y in range(grid_size):
                        for x in range(grid_size):
                            if movement_stats[y, x] > 0:
                                grid_key = f"grid_{y}_{x}"
                                movement_path_data[grid_key] = {
                                    'count': int(movement_stats[y, x]),
                                    'intensity': float(normalized_movement[y, x]),
                                    'x': x * grid_width + grid_width // 2,
                                    'y': y * grid_height + grid_height // 2
                                }
                
                # 최종 히트맵 저장
                final_heatmap_path = os.path.join(app.config['PROCESSED_FOLDER'], 'heatmaps', f"{upload_id}_final_heatmap_{timestamp}.jpg")
                final_heatmap_image = self._create_heatmap_overlay(heatmap, background.copy())
                cv2.imwrite(final_heatmap_path, final_heatmap_image)
                
                # 분석 결과 업데이트
                analysis.analysis_date = datetime.now()
                analysis.duration = duration
                analysis.frame_count = frame_count
                analysis.total_persons = total_persons
                analysis.max_persons = max_persons_in_frame
                analysis.processed_video_path = processed_video_path
                analysis.heatmap_video_path = heatmap_video_path
                analysis.bounding_box_video_path = bounding_box_video_path
                analysis.heatmap_path = final_heatmap_path
                analysis.set_frame_data(frame_data_list)
                analysis.set_time_stats(time_stats)
                analysis.set_movement_data(movement_path_data)
                analysis.status = 'completed'
                db.session.commit()
                
                self.logger.info(f"비디오 처리 완료: {upload_id}")
                
            except Exception as e:
                self.logger.error(f"비디오 처리 중 오류 발생: {e}")
                with app.app_context():
                    analysis = VideoAnalysis.query.filter_by(upload_id=upload_id).first()
                    if analysis:
                        analysis.status = 'failed'
                        analysis.error_message = str(e)
                        db.session.commit()
    
    def _create_heatmap_overlay(self, heatmap, background):
        """
        히트맵 오버레이 이미지 생성
        Args:
            heatmap: 히트맵 데이터
            background: 배경 이미지
        Returns:
            히트맵이 오버레이된 이미지
        """
        # 원본 이미지 크기 가져오기
        h, w = background.shape[:2]
        
        # 히트맵 정규화
        if np.max(heatmap) > 0:
            normalized_heatmap = heatmap / np.max(heatmap)
        else:
            normalized_heatmap = heatmap
        
        # 히트맵 색상 매핑
        colormap = plt.cm.jet
        heatmap_colored = (colormap(normalized_heatmap) * 255).astype(np.uint8)
        
        # 알파 채널 조정
        alpha = np.clip(normalized_heatmap * 8, 0, 0.8)  # 알파값 범위 조정
        alpha_channel = (alpha * 255).astype(np.uint8)
        
        background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        
        # 배경에 히트맵 오버레이
        for c in range(3):
            background_rgb[:, :, c] = np.where(
                alpha_channel > 0,
                (1 - alpha[:, :, np.newaxis])[:, :, 0] * background_rgb[:, :, c] + 
                alpha[:, :, np.newaxis][:, :, 0] * heatmap_colored[:, :, c],
                background_rgb[:, :, c]
            )
        
        # RGB -> BGR 변환
        final_image = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2BGR)
        
        # 이미지 선명도 개선
        final_image = cv2.addWeighted(final_image, 1.2, final_image, 0, 0)
    
        detected_count = np.count_nonzero(alpha_channel > 10)  # 낮은 값은 제외
        
        cv2.rectangle(final_image, (5, 5), (400, 45), (220, 220, 220), -1)  # 밝은 회색 배경
        
        # 텍스트 - 현재 프레임의 사람 수
        cv2.putText(final_image, f"Current Frame: {detected_count} persons", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # 검정색 텍스트
        
        scale_factor = 1.15
        new_width = int(w * scale_factor)
        new_height = int(h * scale_factor)
        
        # 이미지 크기 조정
        resized_image = cv2.resize(final_image, (new_width, new_height))
        
        start_x = (new_width - w) // 2
        start_y = (new_height - h) // 2
        final_resized = resized_image[start_y:start_y+h, start_x:start_x+w]
        
        return final_resized
    
    def _save_heatmap_image(self, heatmap, background, output_path):
        """
        히트맵 이미지 저장
        Args:
            heatmap: 히트맵 데이터
            background: 배경 이미지
            output_path: 출력 파일 경로
        """
        final_image = self._create_heatmap_overlay(heatmap, background)
        cv2.imwrite(output_path, final_image) 