"""
설정 파일
"""
import os
from typing import Optional


class Config:
    """기본 설정"""
    
    # FastAPI 설정
    API_TITLE = "Clothing Recommendation AI Service"
    API_VERSION = "0.1.0"
    API_DESCRIPTION = "체형 기반 의류 추천 AI 서비스"
    
    # 서버 설정
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8001))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # 모델 설정
    POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", None)
    
    # Marqo 설정 (로컬 인스턴스 사용 - URL 불필요)
    # MARQO_URL = os.getenv("MARQO_URL", "http://localhost:8000")
    MARQO_INDEX_NAME = os.getenv("MARQO_INDEX_NAME", "fashion_sigclip")
    
    # CORS 설정
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    CORS_CREDENTIALS = True
    CORS_METHODS = ["*"]
    CORS_HEADERS = ["*"]
    
    # 로깅 설정
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class DevelopmentConfig(Config):
    """개발 환경 설정"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """운영 환경 설정"""
    DEBUG = False
    LOG_LEVEL = "INFO"


def get_config() -> Config:
    """환경에 따라 설정 반환"""
    env = os.getenv("ENV", "development")
    
    if env == "production":
        return ProductionConfig()
    else:
        return DevelopmentConfig()
