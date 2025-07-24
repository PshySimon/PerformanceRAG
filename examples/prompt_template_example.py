"""
Prompt 模板使用示例
"""

from utils.prompt import (
    create_template, fill_template, get_template, quick_fill,
    list_available_templates, create_rag_prompt, create_multi_turn_prompt,
    prompt_manager
)


def basic_usage_example():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 1. 创建简单模板
    template = create_template("你好，{name}！今天是{date}。")
    result = template.fill(name="张三", date="2024-01-01")
    print(f"简单填充: {result}")
    
    # 2. 使用便捷函数
    result2 = fill_template("问题：{question}\n答案：{answer}", 
                           question="什么是AI？", answer="人工智能")
    print(f"便捷填充: {result2}")
    
    # 3. 查看模板变量
    print(f"模板变量: {template.variables}")


def predefined_template_example():
    """预定义模板示例"""
    print("\n=== 预定义模板示例 ===")
    
    # 1. 列出所有可用模板
    templates = list_available_templates()
    print(f"可用模板: {templates[:5]}...")  # 只显示前5个
    
    # 2. 使用预定义模板
    result = quick_fill("question_answer", 
                       question="Python是什么？", 
                       answer="一种编程语言")
    print(f"问答模板: {result}")
    
    # 3. 使用查询扩展模板
    result2 = quick_fill("rewrite", query="如何学习机器学习")
    print(f"查询重写模板: {result2}")


def rag_example():
    """RAG 相关示例"""
    print("\n=== RAG 示例 ===")
    
    # 模拟检索到的文档
    documents = [
        "Python是一种高级编程语言，由Guido van Rossum创建。",
        "Python具有简洁的语法和强大的库生态系统。",
        "Python广泛应用于数据科学、Web开发和人工智能领域。"
    ]
    
    question = "Python有什么特点？"
    
    # 创建RAG提示词
    rag_prompt = create_rag_prompt(question, documents)
    print(f"RAG提示词:\n{rag_prompt}")


def advanced_usage_example():
    """高级使用示例"""
    print("\n=== 高级使用示例 ===")
    
    # 1. 部分填充
    template = get_template("context_question")
    partial = template.partial_fill(context="这是一个关于AI的文档...")
    result = partial.fill(question="AI的应用领域有哪些？")
    print(f"部分填充结果: {result}")
    
    # 2. 模板验证
    is_valid = template.validate(context="测试", question="测试问题")
    print(f"模板验证: {is_valid}")
    
    # 3. 注册自定义模板
    prompt_manager.register_template(
        "custom_template",
        "这是一个自定义模板：{param1} 和 {param2}"
    )
    
    result = quick_fill("custom_template", param1="参数1", param2="参数2")
    print(f"自定义模板: {result}")


def multi_turn_example():
    """多轮对话示例"""
    print("\n=== 多轮对话示例 ===")
    
    conversation_history = [
        {"role": "user", "content": "什么是机器学习？"},
        {"role": "assistant", "content": "机器学习是人工智能的一个分支..."},
        {"role": "user", "content": "它有哪些类型？"},
        {"role": "assistant", "content": "主要有监督学习、无监督学习和强化学习..."}
    ]
    
    current_question = "能详细解释一下监督学习吗？"
    
    multi_turn_prompt = create_multi_turn_prompt(conversation_history, current_question)
    print(f"多轮对话提示词:\n{multi_turn_prompt}")


if __name__ == "__main__":
    basic_usage_example()
    predefined_template_example()
    rag_example()
    advanced_usage_example()
    multi_turn_example()