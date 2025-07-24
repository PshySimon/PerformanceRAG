from typing import Dict, List

from utils.config import config
from utils.logger import get_logger

from .executor import PipelineExecutor
from .registry import ComponentRegistry


class PipelineBuilder:
    """Pipeline构建器，负责根据配置构建Pipeline"""

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.logger = get_logger(__name__)
        self.config_data = None
        self.components = {}
        self.built_pipeline = None

    def load_config(self) -> "PipelineBuilder":
        """从utils.config加载配置"""
        try:
            # 使用utils.config获取配置
            if hasattr(config, self.pipeline_name):
                self.config_data = getattr(config, self.pipeline_name)
                if self.logger:
                    self.logger.debug(f"成功加载Pipeline配置: {self.pipeline_name}")
            else:
                raise ValueError(f"找不到Pipeline配置: {self.pipeline_name}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"加载Pipeline配置失败: {e}")
            raise

        return self

    def validate_config(self) -> "PipelineBuilder":
        """验证配置的完整性"""
        if not self.config_data:
            raise ValueError("配置数据为空，请先调用load_config()")

        # 检查必需的配置项
        required_keys = ["components"]
        for key in required_keys:
            if key not in self.config_data:
                raise ValueError(f"配置中缺少必需的键: {key}")

        # 验证组件配置
        components_config = self.config_data.get("components", {})
        if not components_config:
            raise ValueError("组件配置为空")

        for comp_name, comp_config in components_config.items():
            if "type" not in comp_config:
                raise ValueError(f"组件 {comp_name} 缺少type配置")
            if "name" not in comp_config:
                raise ValueError(f"组件 {comp_name} 缺少name配置")

        # 验证流程配置
        flow_config = self.config_data.get("flow", {})
        if flow_config:
            for comp_name, next_comps in flow_config.items():
                if comp_name not in components_config:
                    raise ValueError(f"流程中引用了不存在的组件: {comp_name}")
                if isinstance(next_comps, list):
                    for next_comp in next_comps:
                        if next_comp not in components_config:
                            raise ValueError(f"流程中引用了不存在的组件: {next_comp}")

        # 验证入口点
        entry_points = self.config_data.get("entry_points", {})
        if entry_points:
            for entry_name, entry_comp in entry_points.items():
                if entry_comp not in components_config:
                    raise ValueError(
                        f"入口点 {entry_name} 引用了不存在的组件: {entry_comp}"
                    )

        if self.logger:
            self.logger.debug(f"Pipeline配置验证通过: {self.pipeline_name}")

        return self

    def create_components(self) -> "PipelineBuilder":
        """创建所有组件实例"""
        if not self.config_data:
            raise ValueError("配置数据为空，请先调用load_config()")

        components_config = self.config_data.get("components", {})

        for comp_name, comp_config in components_config.items():
            try:
                # 获取组件类型和名称
                component_type = comp_config.get("type")
                component_name = comp_config.get("name")
                component_config = comp_config.get("config", {})

                # 从注册表获取组件类
                component_class = ComponentRegistry.get(component_type, component_name)

                # 创建组件实例
                self.components[comp_name] = component_class(
                    comp_name, component_config
                )

                if self.logger:
                    self.logger.debug(
                        f"创建组件: {comp_name} ({component_type}.{component_name})"
                    )

            except Exception as e:
                if self.logger:
                    self.logger.error(f"创建组件 {comp_name} 失败: {e}")
                raise

        return self

    def connect_components(self) -> "PipelineBuilder":
        """连接组件，构建流程图"""
        flow_config = self.config_data.get("flow", {})

        if not flow_config:
            if self.logger:
                self.logger.warning("没有配置组件流程")
            return self

        for comp_name, next_comps in flow_config.items():
            if comp_name not in self.components:
                raise ValueError(f"未找到组件: {comp_name}")

            current_comp = self.components[comp_name]

            # 处理下一步组件（可能是单个组件或组件列表）
            if isinstance(next_comps, str):
                next_comps = [next_comps]
            elif not isinstance(next_comps, list):
                raise ValueError(f"组件 {comp_name} 的下一步配置格式错误")

            for next_comp_name in next_comps:
                if next_comp_name not in self.components:
                    raise ValueError(f"未找到组件: {next_comp_name}")
                current_comp.add_next(self.components[next_comp_name])

                if self.logger:
                    self.logger.debug(f"连接组件: {comp_name} -> {next_comp_name}")

        return self

    def initialize_components(self) -> "PipelineBuilder":
        """初始化所有组件"""
        for comp_name, component in self.components.items():
            try:
                component.initialize()
                if self.logger:
                    self.logger.debug(f"初始化组件: {comp_name}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"初始化组件 {comp_name} 失败: {e}")
                raise

        return self

    def build(self) -> PipelineExecutor:
        """构建完整的Pipeline"""
        if self.built_pipeline:
            return self.built_pipeline

        try:
            # 执行构建流程
            self.load_config()
            self.validate_config()
            self.create_components()
            self.connect_components()
            self.initialize_components()

            # 创建PipelineExecutor
            self.built_pipeline = PipelineExecutor(self.config_data)
            self.built_pipeline.components = self.components
            self.built_pipeline._built = True

            # 设置入口点
            entry_points = self.config_data.get("entry_points", {})
            if entry_points:
                # 如果有多个入口点，默认使用第一个
                default_entry = list(entry_points.values())[0]
                if default_entry in self.components:
                    self.built_pipeline.entry_point = self.components[default_entry]
            else:
                # 如果没有配置入口点，尝试使用第一个组件
                if self.components:
                    first_comp_name = list(self.components.keys())[0]
                    self.built_pipeline.entry_point = self.components[first_comp_name]

            if self.logger:
                self.logger.info(f"Pipeline构建完成: {self.pipeline_name}")

            return self.built_pipeline

        except Exception as e:
            if self.logger:
                self.logger.error(f"Pipeline构建失败: {e}")
            raise

    @classmethod
    def from_config(cls, pipeline_name: str) -> "PipelineBuilder":
        """从配置创建Builder实例"""
        return cls(pipeline_name)

    def get_component(self, comp_name: str):
        """获取指定组件"""
        return self.components.get(comp_name)

    def list_components(self) -> List[str]:
        """列出所有组件名称"""
        return list(self.components.keys())

    def get_entry_points(self) -> Dict[str, str]:
        """获取所有入口点"""
        return self.config_data.get("entry_points", {}) if self.config_data else {}
