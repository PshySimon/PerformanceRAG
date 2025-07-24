import os
import yaml


class _AttrDict(dict):
    """支持通过 . 属性访问嵌套 config"""

    def __getattr__(self, name):
        if name in self:
            value = self[name]
            if isinstance(value, dict):
                value = _AttrDict(value)
                self[name] = value
            return value
        raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        self[name] = value


class Config:
    """全局 config 实例，加载 config/ 下所有 yaml 文件"""

    def __init__(self, config_dir: str = None):
        self._data = _AttrDict()
        self._config_dir = config_dir or os.path.join(
            os.path.dirname(__file__), "../config"
        )
        self._load_all()

    def _load_all(self):
        if not os.path.isdir(self._config_dir):
            return
        for fname in os.listdir(self._config_dir):
            if fname.endswith((".yaml", ".yml")):
                key = os.path.splitext(fname)[0]
                path = os.path.join(self._config_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                self._data[key] = _AttrDict(data)

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"No such config: {name}")

    def __repr__(self):
        return f"<Config {list(self._data.keys())}>"


# 单例
config = Config()
