"""
FastAPI 기반 의류 추천 AI 서비스 메인 서버

처리 흐름:
1. /recommend POST: MediaPipe 관절값 + 태그 받기
2. 관절값 → beta 값 변환 (AI 모델)
3. beta값 + 태그 → 프롬프트 생성
4. 프롬프트 → Marqo 검색 → 의류 추천 반환
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import get_config
from models.schemas import PoseInput, RecommendationResult
from services.pose_to_beta import get_converter
from services.profile_generator import get_generator
from services.marqo_service import get_recommender


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 설정 로드
config = get_config()

# FastAPI 앱 초기화
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=config.CORS_CREDENTIALS,
    allow_methods=config.CORS_METHODS,
    allow_headers=config.CORS_HEADERS,
)


# ============================================================================
# 라이프사이클 이벤트
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    logger.info("=" * 60)
    logger.info("Clothing Recommendation AI Service Started")
    logger.info("=" * 60)
    
    # 각 서비스 초기화 확인
    try:
        converter = get_converter(config.POSE_MODEL_PATH)
        logger.info("✓ Pose to Beta converter initialized")
    except Exception as e:
        logger.error(f"✗ Pose to Beta converter initialization failed: {e}")
    
    try:
        generator = get_generator()
        logger.info("✓ Profile generator initialized")
    except Exception as e:
        logger.error(f"✗ Profile generator initialization failed: {e}")
    
    try:
        recommender = get_recommender()
        if recommender.health_check():
            logger.info("✓ Marqo recommender initialized")
        else:
            logger.warning("⚠ Marqo initialization failed - will use mock results")
    except Exception as e:
        logger.error(f"✗ Marqo recommender initialization failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    logger.info("Clothing Recommendation AI Service Shutting Down")


# ============================================================================
# 헬스 체크
# ============================================================================

@app.get("/health")
async def health_check():
    """서비스 상태 확인"""
    return {
        "status": "healthy",
        "service": "clothing-recommendation-ai",
        "version": config.API_VERSION
    }


# ============================================================================
# 메인 API: 의류 추천
# ============================================================================

@app.post("/recommend", response_model=RecommendationResult)
async def recommend_clothing(request: PoseInput):
    """
    의류 추천 API
    
    처리 과정:
    1. MediaPipe 관절값 → SMPL beta 값 변환
    2. beta 값 + 태그 → SigLiP 프롬프트 생성
    3. 프롬프트 → Marqo 검색 → 추천 의류 반환
    
    Args:
        request: PoseInput
            - user_id: 사용자 ID
            - joints: MediaPipe 관절값
            - tags: 의류 추천 태그
            - metadata: 추가 정보
    
    Returns:
        RecommendationResult
            - user_id: 사용자 ID
            - recommendations: 추천 의류 목록
            - profile: 생성된 프로필 (beta값, 프롬프트 등)
            - metadata: 추가 정보
    """
    
    logger.info(f"Recommendation requested for user: {request.user_id}")
    
    try:
        # ====== Stage 1: 관절값 → beta 값 변환 ======
        logger.info(f"Stage 1: Converting pose to beta values...")
        converter = get_converter(config.POSE_MODEL_PATH)
        beta_result = converter.predict_beta(request.joints)
        beta_values = beta_result['beta_values']
        confidence = beta_result.get('confidence', 0.0)
        
        logger.info(
            f"  - Beta values generated (confidence: {confidence:.2f})"
        )
        
        
        # ====== Stage 2 & 3: beta + 태그 → 프롬프트 생성 ======
        logger.info(f"Stage 2-3: Generating SigLiP prompt...")
        generator = get_generator()
        prompt_result = generator.generate_prompt(beta_values, request.tags)
        generated_prompt = prompt_result['prompt']
        prompt_components = prompt_result['components']
        
        logger.info(f"  - Prompt generated successfully")
        
        
        # ====== Stage 4: 프롬프트 → Marqo 검색 ======
        logger.info(f"Stage 4: Searching recommendations via Marqo...")
        recommender = get_recommender()
        recommendations = recommender.search_recommendations(
            prompt=generated_prompt,
            top_k=10,
            filter_tags=None  # 나중에 필터링 추가 가능
        )
        
        logger.info(f"  - {len(recommendations)} recommendations found")
        
        
        # ====== 결과 조합 ======
        from models.schemas import PromptProfile
        
        profile = PromptProfile(
            user_id=request.user_id,
            beta_values=beta_values,
            tags=request.tags,
            generated_prompt=generated_prompt,
            prompt_components=prompt_components
        )
        
        result = RecommendationResult(
            user_id=request.user_id,
            recommendations=recommendations,
            profile=profile,
            metadata={
                'beta_confidence': confidence,
                'marqo_status': 'success' if recommender.client else 'mock'
            }
        )
        
        logger.info(f"✓ Recommendation completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"✗ Error in recommendation pipeline: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}"
        )


# ============================================================================
# 디버그 API: 각 단계별 결과 확인
# ============================================================================

@app.post("/debug/pose-to-beta")
async def debug_pose_to_beta(request: PoseInput):
    """Stage 1 디버그: 관절값 → beta 값"""
    try:
        converter = get_converter(config.POSE_MODEL_PATH)
        result = converter.predict_beta(request.joints)
        return {
            'user_id': request.user_id,
            'result': result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/prompt-generation")
async def debug_prompt_generation(request: PoseInput):
    """Stage 2-3 디버그: beta + 태그 → 프롬프트"""
    try:
        converter = get_converter(config.POSE_MODEL_PATH)
        beta_result = converter.predict_beta(request.joints)
        beta_values = beta_result['beta_values']
        
        generator = get_generator()
        prompt_result = generator.generate_prompt(beta_values, request.tags)
        
        return {
            'user_id': request.user_id,
            'beta_values': beta_values,
            'prompt_result': prompt_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/marqo-search")
async def debug_marqo_search(prompt: str, top_k: int = 10):
    """Stage 4 디버그: 프롬프트 → 검색"""
    try:
        recommender = get_recommender()
        results = recommender.search_recommendations(prompt, top_k)
        
        return {
            'prompt': prompt,
            'results': results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 메타데이터 API
# ============================================================================

@app.get("/info")
async def get_info():
    """서비스 정보"""
    return {
        'title': config.API_TITLE,
        'version': config.API_VERSION,
        'description': config.API_DESCRIPTION,
        'marqo': 'local_instance',
        'debug': config.DEBUG
    }


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG
    )
