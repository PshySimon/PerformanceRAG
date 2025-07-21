# Query Expansion Prompts

REWRITE_PROMPT = (
    "请将下列用户查询改写为更适合检索的表达，直接输出改写内容，格式为<output>改写内容</output>：\n"
    "原始查询：{query}"
)

HYDE_PROMPT = (
    "请根据下列查询生成一段与之相关的假想答案（HyDE），直接输出内容，格式为<output>内容</output>：\n"
    "查询：{query}"
)

MULTI_QUERY_PROMPT = (
    "请为下列查询生成{n}个不同但相关的检索表达，每行一个，格式为<output>表达1\n表达2\n...\n表达N</output>：\n"
    "查询：{query}"
)

DECOMPOSE_PROMPT = (
    "请将下列复杂查询拆解为若干简单子查询，每行一个，格式为<output>子查询1\n子查询2\n...</output>：\n"
    "复杂查询：{query}"
)

DISAMBIGUATE_PROMPT = (
    "请对下列查询进行消歧，补充上下文使其更明确，直接输出消歧后内容，格式为<output>消歧内容</output>：\n"
    "原始查询：{query}"
)

ABSTRACT_PROMPT = (
    "请将下列具体查询抽象为更通用的问题，直接输出抽象后内容，格式为<output>抽象内容</output>：\n"
    "具体查询：{query}"
) 