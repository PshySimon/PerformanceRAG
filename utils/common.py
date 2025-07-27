"""
常量定义文件
存放各种模板和常量
"""

# ==================== Prompt 模板常量 ====================

# 基础模板
BASIC_TEMPLATES = {
    "question_answer": "问题：{question}\n答案：{answer}",
    "context_question": "上下文：{context}\n\n问题：{question}\n\n请基于上下文回答问题。",
    "system_user": "系统：{system}\n\n用户：{user}",
}

# RAG 相关模板
RAG_TEMPLATES = {
    "retrieval_prompt": (
        "你是一名知识问答助手，仅根据下方检索到的参考资料来回答用户问题。请遵循以下严格的规则：\n"
        "请严格遵循以下要求：\n"
        "1. 你的回答必须仅基于提供的“参考资料”内容；\n"
        "2. 如果“参考资料”中没有与问题相关的信息，请直接回答：“抱歉，我暂时还没学会这些知识。” 不要编造、猜测或引用未提供的内容；\n"
        "3. 回答要准确、简洁，不添加主观判断或额外补充说明；\n"
        "4. 禁止加入模型自身观点、解释过程或语言风格润色，只需客观作答。\n"
        "5. 你回答的语气应该专业、柔和，把检索到的知识当做是你已经理解的知识作答\n"
        "---\n参考资料：\n{documents}\n---\n"
        "---\n用户问题：{question}\n---\n"
        "请根据上述要求，给出规范且可靠的回答："
    ),
    "rerank_prompt": (
        "请对以下文档片段按照与问题的相关性进行排序：\n\n"
        "问题：{question}\n\n"
        "文档片段：\n{documents}\n\n"
        "请按相关性从高到低排序，输出序号列表："
    ),
    "summary_prompt": (
        "请总结以下内容：\n\n" "{content}\n\n" "总结要求：{requirements}"
    ),
}

# 查询扩展模板（继承现有的）
QUERY_EXPANSION_TEMPLATES = {
    "rewrite": (
        "请将下列用户查询改写为更适合检索的表达，直接输出改写内容，格式为<output>改写内容</output>：\n"
        "原始查询：{query}"
    ),
    "hyde": (
        "请根据下列查询生成一段与之相关的假想答案（HyDE），直接输出内容，格式为<output>内容</output>：\n"
        "查询：{query}"
    ),
    "multi_query": (
        "请为下列查询生成{n}个不同但相关的检索表达，每行一个，格式为<output>表达1\n表达2\n...\n表达N</output>：\n"
        "查询：{query}"
    ),
    "decompose": (
        "请将下列复杂查询拆解为若干简单子查询，每行一个，格式为<output>子查询1\n子查询2\n...</output>：\n"
        "复杂查询：{query}"
    ),
    "disambiguate": (
        "请对下列查询进行消歧，补充上下文使其更明确，直接输出消歧后内容，格式为<output>消歧内容</output>：\n"
        "原始查询：{query}"
    ),
    "abstract": (
        "请将下列具体查询抽象为更通用的问题，直接输出抽象后内容，格式为<output>抽象内容</output>：\n"
        "具体查询：{query}"
    ),
}

# 评估模板
EVALUATION_TEMPLATES = {
    "relevance_score": (
        "请评估以下答案与问题的相关性，打分0-10：\n\n"
        "问题：{question}\n\n"
        "答案：{answer}\n\n"
        "评分标准：{criteria}\n\n"
        "请给出分数和理由："
    ),
    "quality_assessment": (
        "请评估以下内容的质量：\n\n"
        "内容：{content}\n\n"
        "评估维度：{dimensions}\n\n"
        "请提供详细评估："
    ),
}

# 所有模板的集合
ALL_TEMPLATES = {
    **BASIC_TEMPLATES,
    **RAG_TEMPLATES,
    **QUERY_EXPANSION_TEMPLATES,
    **EVALUATION_TEMPLATES,
}

# ==================== 其他常量 ====================

# 默认配置
DEFAULT_CONFIG = {
    "max_length": 4096,
    "temperature": 0.7,
    "top_p": 0.9,
}

# 文件类型常量
SUPPORTED_FILE_TYPES = [".txt", ".md", ".pdf", ".docx", ".html", ".json", ".csv"]

# 语言代码
LANGUAGE_CODES = {
    "中文": "zh",
    "英文": "en",
    "日文": "ja",
    "韩文": "ko",
}
