"""
데이터 스키마 정의
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class Tag(BaseModel):
    """태그 정보"""
    season: Optional[str] = Field(None, description="계절 (봄, 여름, 가을, 겨울)")
    style: Optional[str] = Field(None, description="분위기/스타일 (캐주얼, 포멀, 스포츠 등)")
    color: Optional[str] = Field(None, description="선호 색상")
    category: Optional[str] = Field(None, description="의류 부위 (상의, 하의, 아우터 등)")
    additional_tags: Optional[List[str]] = Field(None, description="추가 태그")


class PoseInput(BaseModel):
    """백엔드에서 받을 입력 데이터"""
    user_id: str = Field(..., description="사용자 ID")
    joints: List[float] = Field(..., description="MediaPipe 관절값 (33개 또는 필요한 수만큼)")
    tags: Tag = Field(..., description="의류 추천 태그")
    metadata: Optional[Dict] = Field(None, description="추가 메타데이터")


class BetaOutput(BaseModel):
    """AI 모델 출력 (beta 값)"""
    user_id: str
    beta_values: List[float] = Field(..., min_items=10, max_items=10, description="SMPL beta 값 (체형 파라미터)")
    confidence: Optional[float] = Field(None, description="예측 신뢰도")


class PromptProfile(BaseModel):
    """SigLiP용 프롬프트 프로필"""
    user_id: str
    beta_values: List[float]
    tags: Tag
    generated_prompt: str = Field(..., description="생성된 프롬프트")
    prompt_components: Dict = Field(..., description="프롬프트 구성 요소")


class RecommendationResult(BaseModel):
    """최종 추천 결과"""
    user_id: str
    recommendations: List[Dict] = Field(..., description="추천 의류 반환 결과 (Marqo)")
    profile: PromptProfile
    metadata: Optional[Dict] = Field(None, description="추가 정보")
