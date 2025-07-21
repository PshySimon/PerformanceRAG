from utils.config import config
from utils.llm.openai_llm import OpenAILLM
from utils.llm.zhipu_llm import ZhipuLLM

class LLMFactory:
    @staticmethod
    def from_config():
        llm_config = config.llm.clients[config.llm.default]
        llm_type = llm_config.get("type")
        if llm_type == "openai":
            return OpenAILLM(**llm_config)
        elif llm_type == "zhipu":
            return ZhipuLLM(**llm_config)
        else:
            raise ValueError(f"不支持的LLM类型: {llm_type}") 