"""
Stage 2 & 3: beta값 + 태그 → SigLiP용 프롬프트 생성
"""
from typing import Dict, List, Optional
from ..models.schemas import Tag
import logging

logger = logging.getLogger(__name__)


class ProfileGenerator:
    """
    SMPL beta값과 사용자 태그를 기반으로 
    SigLiP 모델에 넣을 프롬프트 생성
    """
    
    def __init__(self):
        """프롬프트 생성 규칙 초기화"""
        self.season_keywords = {
            '봄': ['라이트', '파스텔', '신선한'],
            '여름': ['밝은', '통풍이 잘 되는', '시원한'],
            '가을': ['따뜻한', '어두운', '절제된'],
            '겨울': ['두꺼운', '따뜻한', '어두운']
        }
        
        self.style_keywords = {
            '캐주얼': ['편한', '실용적인', '일상적인'],
            '포멀': ['정장', '전문적인', '세련된'],
            '스포츠': ['활동적인', '성능 중심', '에너지적인'],
            '보헤미안': ['자유로운', '예술적인', '유니크한'],
            '미니멀': ['심플한', '필수적인', '깔끔한']
        }
    
    def _interpret_beta_values(self, beta_values: List[float]) -> Dict[str, str]:
        """
        beta 값을 해석하여 체형 특징으로 변환
        
        SMPL beta 파라미터 해석 (참고):
        - beta[0]: 전체 크기/체중
        - beta[1]: 길쭉함/뚱뚱함
        - beta[2}-beta[9]: 세부 체형 특징들
        """
        body_features = {}
        
        # 전체 크기 (beta[0])
        if beta_values[0] > 0.5:
            body_features['size'] = '큰'
        elif beta_values[0] < -0.5:
            body_features['size'] = '작은'
        else:
            body_features['size'] = '중간'
        
        # 길쭉함 vs 뚱뚱함 (beta[1])
        if beta_values[1] > 0.5:
            body_features['build'] = '뚱뚱한'
        elif beta_values[1] < -0.5:
            body_features['build'] = '길쭉한'
        else:
            body_features['build'] = '균형잡힌'
        
        # 상체/하체 비율 (beta[2])
        if beta_values[2] > 0.5:
            body_features['torso'] = '상체가 큰'
        elif beta_values[2] < -0.5:
            body_features['torso'] = '하체가 큰'
        else:
            body_features['torso'] = '균형잡힌 체형'
        
        # 어깨 너비 (beta[3])
        if beta_values[3] > 0.5:
            body_features['shoulders'] = '넓은 어깨'
        elif beta_values[3] < -0.5:
            body_features['shoulders'] = '좁은 어깨'
        else:
            body_features['shoulders'] = '일반적인 어깨'
        
        return body_features
    
    def _build_tag_keywords(self, tags: Tag) -> List[str]:
        """태그에서 키워드 추출"""
        keywords = []
        
        if tags.season:
            keywords.extend(self.season_keywords.get(tags.season, []))
        
        if tags.style:
            keywords.extend(self.style_keywords.get(tags.style, []))
        
        if tags.color:
            keywords.append(f"{tags.color}색")
        
        if tags.category:
            keywords.append(tags.category)
        
        if tags.additional_tags:
            keywords.extend(tags.additional_tags)
        
        return keywords
    
    def generate_prompt(
        self,
        beta_values: List[float],
        tags: Tag
    ) -> Dict:
        """
        프롬프트 생성
        
        Args:
            beta_values: SMPL beta 값 (10개)
            tags: 사용자 태그
            
        Returns:
            {
                'prompt': 생성된 프롬프트 텍스트,
                'components': {
                    'body_features': 체형 특징,
                    'tag_keywords': 태그 키워드,
                    'prompt_template': 사용된 템플릿
                }
            }
        """
        # 체형 특징 해석
        body_features = self._interpret_beta_values(beta_values)
        
        # 태그 키워드 추출
        tag_keywords = self._build_tag_keywords(tags)
        
        # 프롬프트 생성
        prompt_parts = []
        
        # 기본 프롬프트
        prompt_parts.append("다음 조건에 맞는 의류를 찾아주세요:")
        
        # 체형 정보 추가
        prompt_parts.append(f"\n체형 특징:")
        for key, value in body_features.items():
            prompt_parts.append(f"  - {key}: {value}")
        
        # 스타일 정보 추가
        if tag_keywords:
            prompt_parts.append(f"\n선호사항: {', '.join(tag_keywords)}")
        
        # 세부 요청사항 추가
        if tags.category:
            prompt_parts.append(f"\n의류: {tags.category}")
        
        generated_prompt = "\n".join(prompt_parts)
        
        return {
            'prompt': generated_prompt,
            'components': {
                'body_features': body_features,
                'tag_keywords': tag_keywords,
                'prompt_template': 'basic_recommendation',
                'raw_beta_values': beta_values
            }
        }


# 전역 변수로 생성기 초기화
_generator: Optional[ProfileGenerator] = None


def get_generator() -> ProfileGenerator:
    """생성기 인스턴스 가져오기 (싱글톤 패턴)"""
    global _generator
    if _generator is None:
        _generator = ProfileGenerator()
    return _generator
