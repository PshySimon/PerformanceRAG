from typing import Any, Dict, List
from .base_query import BaseQueryComponent
from utils.prompt import quick_fill
from rag.pipeline.registry import ComponentRegistry


@ComponentRegistry.register('query', 'decomposition')
class DecompositionComponent(BaseQueryComponent):
    """查询分解组件 - 将复杂查询分解为简单子查询"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.max_subqueries = config.get('max_subqueries', 5)
        self.min_subqueries = config.get('min_subqueries', 2)
        
    def _optimize_query(self, query: str) -> Dict[str, Any]:
        """执行查询分解"""
        try:
            # 首先判断是否需要分解
            if not self._needs_decomposition(query):
                return {
                    'subqueries': [query],
                    'decomposition_needed': False,
                    'original_query': query
                }
            
            # 执行分解
            prompt = quick_fill('decompose', query=query)
            response = self._call_llm_with_retry(prompt)
            content = self._extract_output(response)
            
            # 解析子查询
            subqueries = [line.strip() for line in content.splitlines() if line.strip()]
            
            # 过滤和验证子查询
            subqueries = self._filter_subqueries(subqueries, query)
            
            return {
                'subqueries': subqueries,
                'decomposition_needed': True,
                'original_query': query,
                'num_subqueries': len(subqueries)
            }
            
        except Exception as e:
            self.logger.error(f"查询分解失败: {e}")
            return {'subqueries': [query], 'decomposition_needed': False}
    
    def _needs_decomposition(self, query: str) -> bool:
        """判断查询是否需要分解"""
        # 简单的启发式规则
        decomposition_indicators = [
            '和', '以及', '还有', '另外', '同时', '并且',
            'and', 'also', 'additionally', 'furthermore',
            '？', '?', '如何', 'how', '什么', 'what', '为什么', 'why'
        ]
        
        # 检查查询长度
        if len(query) < 10:
            return False
            
        # 检查是否包含分解指示词
        indicator_count = sum(1 for indicator in decomposition_indicators if indicator in query.lower())
        
        return indicator_count >= 2 or len(query.split()) > 15
    
    def _filter_subqueries(self, subqueries: List[str], original_query: str) -> List[str]:
        """过滤和验证子查询"""
        filtered = []
        
        for subquery in subqueries:
            # 过滤太短或太长的子查询
            if len(subquery) < 5 or len(subquery) > 200:
                continue
                
            # 过滤与原查询完全相同的子查询
            if subquery.lower().strip() == original_query.lower().strip():
                continue
                
            # 过滤重复的子查询
            if subquery not in filtered:
                filtered.append(subquery)
                
            # 限制子查询数量
            if len(filtered) >= self.max_subqueries:
                break
        
        # 如果子查询太少，添加原查询
        if len(filtered) < self.min_subqueries:
            filtered.insert(0, original_query)
            
        return filtered