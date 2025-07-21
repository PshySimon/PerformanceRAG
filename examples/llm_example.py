from utils.llm import LLMFactory
from utils.config import config

if __name__ == "__main__":
    # 选择默认模型
    llm = LLMFactory.from_config()

    # chat接口演示
    try:
        reply = llm.completion("你好，帮我介绍一下Python。")
        print("[CHAT] 回复：", reply)
    except Exception as e:
        print("[WARN] chat接口调用失败（如无API KEY或网络问题）：", e)

    # completion接口演示
    prompt = "请写一段Python代码，实现斐波那契数列。"
    try:
        result = llm.completion(prompt)
        print("[COMPLETION] 补全：", result)
    except Exception as e:
        print("[WARN] completion接口调用失败（如无API KEY或网络问题）：", e)
