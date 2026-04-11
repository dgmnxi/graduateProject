"""
Stage 4: Marqo-FashionSigLip을 활용한 의류 추천
HuggingFace 모델을 직접 로드하여 텍스트 임베딩 검색
"""
from typing import Dict, List, Optional, Any
import logging
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


class MarqoRecommender:
    """
    프롬프트 기반 의류 추천을 위해 Marqo-FashionSigLIP 모델을 로컬에서 직접 사용
    """
    
    def __init__(self):
        """HuggingFace 모델 초기화"""
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "Marqo/marqo-fashionSigLIP"
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Marqo-FashionSigLIP 모델 로드"""
        try:
            logger.info(f"Loading Marqo-FashionSigLIP model...")
            
            # HuggingFace에서 모델과 프로세서 로드
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # GPU 또는 CPU로 이동
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✓ Marqo-FashionSigLIP model loaded on {self.device}")
            
        except ImportError as e:
            logger.error(
                f"Failed to import required packages: {e}. "
                "Install with: pip install transformers torch pillow"
            )
            self.model = None
            self.processor = None
        except Exception as e:
            logger.error(f"Failed to initialize Marqo model: {e}")
            self.model = None
            self.processor = None
    
    def search_recommendations(
        self,
        prompt: str,
        top_k: int = 10,
        filter_tags: Optional[Dict] = None
    ) -> List[Dict]:
        """
        프롬프트 기반 의류 추천 검색
        
        Args:
            prompt: 생성된 프롬프트 (텍스트 설명)
            top_k: 반환할 추천 개수
            filter_tags: 필터링 태그 (선택사항)
            
        Returns:
            추천 의류 리스트
        """
        
        if self.model is None or self.processor is None:
            logger.warning("Marqo model is not initialized. Returning mock results.")
            return self._mock_recommendations(prompt, top_k)
        
        try:
            # 프롬프트로부터 텍스트 임베딩 생성
            prompt_embedding = self._get_text_embedding(prompt)
            
            logger.info(f"Text embedding generated for prompt: {prompt[:50]}...")
            
            # TODO: 실제 데이터베이스에서 의류 상품 검색
            # 현재는 mock 데이터 사용 (데이터베이스 구축 후 구현)
            recommendations = self._mock_recommendations(prompt, top_k)
            
            # 각 추천 아이템에 prompt_embedding 정보 추가 (나중에 활용 가능)
            for rec in recommendations:
                rec['embedding_info'] = {
                    'prompt': prompt[:50],
                    'embedding_generated': True
                }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Marqo search error: {e}")
            return self._mock_recommendations(prompt, top_k)
    
    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """
        텍스트로부터 임베딩 벡터 생성
        
        Args:
            text: 입력 텍스트
            
        Returns:
            정규화된 텍스트 임베딩 벡터
        """
        try:
            # 프로세서를 통해 텍스트 토큰화
            processed = self.processor(
                text=[text],
                padding='max_length',
                return_tensors="pt"
            )
            
            # 모델의 입력을 GPU/CPU로 이동
            processed = {k: v.to(self.device) for k, v in processed.items()}
            
            # 모델로부터 텍스트 특징 추출
            with torch.no_grad():
                text_features = self.model.get_text_features(
                    processed.get('input_ids'),
                    normalize=True
                )
            
            return text_features
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            raise
    
    def _apply_filters(
        self,
        results: List[Dict],
        filter_tags: Dict
    ) -> List[Dict]:
        """결과에 필터 적용"""
        # TODO: 실제 필터링 로직 구현
        filtered = results
        
        if 'color' in filter_tags:
            filtered = [
                r for r in filtered
                if filter_tags['color'].lower() in r.get('tags', '').lower()
            ]
        
        if 'price_range' in filter_tags:
            min_price, max_price = filter_tags['price_range']
            filtered = [
                r for r in filtered
                if min_price <= r.get('price', 0) <= max_price
            ]
        
        return filtered
    
    def _format_results(self, marqo_results: List[Dict]) -> List[Dict]:
        """Marqo 결과를 표준 형식으로 변환"""
        formatted = []
        
        for result in marqo_results:
            formatted.append({
                'id': result.get('_id'),
                'title': result.get('title'),
                'description': result.get('description'),
                'image_url': result.get('image_url'),
                'tags': result.get('tags'),
                'score': result.get('_score'),
                'metadata': result.get('metadata', {})
            })
        
        return formatted
    
    def _mock_recommendations(
        self,
        prompt: str,
        top_k: int
    ) -> List[Dict]:
        """
        모의 추천 결과 생성 (개발/테스트용)
        """
        mock_results = [
            {
                'id': f'item_{i}',
                'title': f'추천 의류 {i+1}',
                'description': f'프롬프트: {prompt[:30]}...에 맞는 의류',
                'image_url': f'https://example.com/images/item_{i}.jpg',
                'tags': ['추천', '의류'],
                'score': 0.95 - (i * 0.05),
                'metadata': {
                    'source': 'mock',
                    'size': 'M',
                    'price': 50000 + i * 5000
                }
            }
            for i in range(min(top_k, 10))
        ]
        
        return mock_results
    
    def health_check(self) -> bool:
        """Marqo 모델 상태 확인"""
        try:
            if self.model is None or self.processor is None:
                logger.warning("Marqo model is not initialized")
                return False
            
            # 간단한 테스트 임베딩으로 헬스 체크
            test_embedding = self._get_text_embedding("test")
            
            logger.info(f"✓ Marqo health check passed. Embedding shape: {test_embedding.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Marqo health check failed: {e}")
            return False


# 전역 변수로 추천기 초기화
_recommender: Optional[MarqoRecommender] = None


def get_recommender() -> MarqoRecommender:
    """추천기 인스턴스 가져오기 (싱글톤 패턴)"""
    global _recommender
    if _recommender is None:
        _recommender = MarqoRecommender()
    return _recommender
