"""
설정 파일
"""
import os
from typing import Optional
from pathlib import Path


class Config:
    """기본 설정"""

    # 프로젝트/데이터 경로 설정
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = Path(os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data")))
    RAW_DATA_ROOT = DATA_ROOT / "raw"
    PROCESSED_DATA_ROOT = DATA_ROOT / "processed"

    # 데이터셋 분리 경로 템플릿
    AGORA_ROOT = RAW_DATA_ROOT / "agora"
    AGORA_IMAGES_DIR = AGORA_ROOT / "images"
    AGORA_ANNOTATIONS_DIR = AGORA_ROOT / "annotations"
    AGORA_SPLITS_DIR = AGORA_ROOT / "splits"

    D3PW_ROOT = RAW_DATA_ROOT / "3dpw"
    D3PW_IMAGES_DIR = D3PW_ROOT / "images"
    D3PW_ANNOTATIONS_DIR = D3PW_ROOT / "annotations"
    D3PW_SPLITS_DIR = D3PW_ROOT / "splits"
    
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
