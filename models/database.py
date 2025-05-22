from flask_sqlalchemy import SQLAlchemy
import logging

logger = logging.getLogger(__name__)
db = SQLAlchemy()

def init_db(app):
    """데이터베이스 연결 및 테이블 초기화"""
    db.init_app(app)
    
    with app.app_context():
        try:
            db.create_all()
            logger.info("데이터베이스 테이블이 성공적으로 생성되었습니다.")
        except Exception as e:
            logger.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
            raise 