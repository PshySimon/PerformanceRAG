from typing import Dict, Any, Optional
from .registry import ComponentRegistry
from utils.logger import get_logger

class PipelineExecutor:
    """Pipeline执行器，负责执行整个流程"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.components = {}
        self.entry_point = None
        self._built = False
        self.logger = get_logger(__name__)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineExecutor':
        """从YAML文件加载配置并创建执行器（已弃用，建议使用Builder）"""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return cls(config)
    
    def build(self):
        """构建Pipeline组件图（已弃用，建议使用Builder）"""
        if self._built:
            return self
        
        # 创建所有组件实例
        components_config = self.config.get('components', {})
        for comp_name, comp_config in components_config.items():
            comp_type = comp_config.get('type')
            comp_name_in_registry = comp_config.get('name')
            
            if not comp_type or not comp_name_in_registry:
                raise ValueError(f"组件配置错误: {comp_name}")
            
            component_class = ComponentRegistry.get(comp_type, comp_name_in_registry)
            self.components[comp_name] = component_class(comp_name, comp_config.get('config', {}))
        
        # 连接组件
        flow_config = self.config.get('flow', {})
        for comp_name, next_comps in flow_config.items():
            if comp_name not in self.components:
                raise ValueError(f"未找到组件: {comp_name}")
            
            current_comp = self.components[comp_name]
            if isinstance(next_comps, str):
                next_comps = [next_comps]
                
            for next_comp_name in next_comps:
                if next_comp_name not in self.components:
                    raise ValueError(f"未找到组件: {next_comp_name}")
                current_comp.add_next(self.components[next_comp_name])
        
        # 设置入口点
        entry_points = self.config.get('entry_points', {})
        if entry_points:
            default_entry = list(entry_points.values())[0]
            if default_entry in self.components:
                self.entry_point = self.components[default_entry]
        else:
            # 使用第一个组件作为入口点
            if self.components:
                first_comp_name = list(self.components.keys())[0]
                self.entry_point = self.components[first_comp_name]
        
        if not self.entry_point:
            raise ValueError("无法确定Pipeline入口点")
        
        # 初始化所有组件
        for component in self.components.values():
            component.initialize()
        
        self._built = True
        return self
    
    def run(self, input_data: Dict[str, Any] = None, entry_point: Optional[str] = None) -> Dict[str, Any]:
        """执行整个Pipeline"""
        if not self._built:
            self.build()
        
        # 选择入口点
        if entry_point and entry_point in self.components:
            start_component = self.components[entry_point]
        else:
            start_component = self.entry_point
            
        if not start_component:
            raise ValueError("无法确定执行入口点")
            
        if self.logger:
            self.logger.debug(f"开始执行Pipeline，入口点: {start_component.name}")
            
        try:
            result = start_component.execute(input_data or {})
            if self.logger:
                self.logger.debug("Pipeline执行完成")
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Pipeline执行失败: {e}")
            raise
    
    def get_component(self, comp_name: str):
        """获取指定组件"""
        return self.components.get(comp_name)
    
    def list_components(self):
        """列出所有组件"""
        return list(self.components.keys())
    
    def get_entry_points(self):
        """获取所有入口点"""
        return self.config.get('entry_points', {})