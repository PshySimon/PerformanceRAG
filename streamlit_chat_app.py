#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Streamlit的RAG对话应用
"""

import os
import sys
import time
from typing import Any, Dict, Generator, List

import streamlit as st

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.pipeline.factory import create_pipeline
from utils.logger import get_logger, setup_logging

# ==================== 配置和初始化 ====================

# 设置页面配置
st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 设置日志
setup_logging(level="DEBUG")
logger = get_logger(__name__)

# 初始化session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False

# ==================== 核心功能函数 ====================


@st.cache_resource
def init_pipeline():
    """初始化pipeline（使用缓存）"""
    try:
        logger.info("初始化ES搜索Pipeline...")
        pipeline = create_pipeline("es_search_pipeline")
        logger.info("Pipeline初始化完成")
        return pipeline, True
    except Exception as e:
        logger.error(f"Pipeline初始化失败: {e}")
        return None, False


def search_with_pipeline(
    query: str, top_k: int, search_type: str, show_intermediate: bool = False
) -> Dict[str, Any]:
    """使用pipeline进行搜索"""
    try:
        if show_intermediate:
            result = st.session_state.pipeline.run_with_intermediate_results(
                {"query": query, "top_k": top_k, "search_type": search_type},
                entry_point="es_retriever",
            )
        else:
            result = st.session_state.pipeline.run(
                {"query": query, "top_k": top_k, "search_type": search_type},
                entry_point="es_retriever",
            )
        return result
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        return {"error": str(e)}


def search_with_pipeline_stream(
    query: str, top_k: int, search_type: str
) -> Generator[Dict[str, Any], None, None]:
    """流式搜索函数，支持状态更新"""
    try:
        pipeline = st.session_state.pipeline
        if not pipeline:
            yield {"error": "Pipeline未初始化"}
            return

        # 1. 开始检索阶段
        yield {
            "current_step": "retrieval",
            "step_state": "running",
            "message": "正在检索相关文档...",
        }
        time.sleep(0.5)

        # 执行实际的pipeline
        result = pipeline.run(
            {"query": query, "top_k": top_k, "search_type": search_type},
            entry_point="es_retriever",
        )

        if "error" in result:
            yield {"error": result["error"]}
            return

        # 2. 检索完成，开始重排
        yield {
            "current_step": "retrieval",
            "step_state": "completed",
            "message": "文档检索完成",
        }
        yield {
            "current_step": "rerank",
            "step_state": "running",
            "message": "正在重新排序文档...",
        }
        time.sleep(0.3)

        # 3. 重排完成，开始生成
        yield {
            "current_step": "rerank",
            "step_state": "completed",
            "message": "文档重排完成",
        }
        yield {
            "current_step": "generation",
            "step_state": "running",
            "message": "正在生成回答...",
        }

        # 4. 模拟流式生成
        answer = result.get("answer", "")
        if answer:
            chunk_size = 10
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i : i + chunk_size]
                partial_answer = answer[: i + len(chunk)]

                yield {
                    "answer_chunk": chunk,
                    "answer_partial": partial_answer,
                    "current_step": "generation",
                    "step_state": "running",
                }
                time.sleep(0.05)

        # 5. 生成完成
        yield {
            "current_step": "generation",
            "step_state": "completed",
            "message": "回答生成完成",
        }
        yield {
            "current_step": "completion",
            "step_state": "completed",
            "message": "处理完成",
        }

        # 6. 返回最终结果
        yield {
            "is_final": True,
            "answer": answer,
            "documents": result.get("documents", []),
            "message": "所有处理完成",
        }

    except Exception as e:
        logger.error(f"流式搜索失败: {e}")
        yield {"error": str(e)}


# ==================== UI组件函数 ====================


def create_progress_indicator():
    """创建简化的进度指示器 - 单行显示当前执行组件"""
    # 创建单个容器用于显示当前状态
    status_container = st.empty()
    
    return {
        "status_container": status_container,
        "current_step": None
    }

def update_status(progress_dict: Dict[str, Any], step: str, status: str, message: str = ""):
    """更新进度状态 - 单行显示"""
    # 状态图标映射
    status_icons = {
        "waiting": "⏳",
        "running": "🔄",  # 转圈效果
        "completed": "✅",
        "error": "❌"
    }
    
    # 步骤名称映射
    step_names = {
        "retrieval": "文档检索",
        "reranking": "结果重排", 
        "generation": "答案生成"
    }
    
    # 只在状态变化时更新显示
    if progress_dict["current_step"] != step or status == "running":
        progress_dict["current_step"] = step
        
        icon = status_icons.get(status, "⏳")
        step_name = step_names.get(step, step)
        
        # 构建显示文本
        if status == "running":
            display_text = f"{icon} 正在执行: **{step_name}**"
            if message:
                display_text += f" - {message}"
        elif status == "completed":
            display_text = f"{icon} 已完成: **{step_name}**"
        elif status == "error":
            display_text = f"{icon} 执行失败: **{step_name}**"
            if message:
                display_text += f" - {message}"
        else:
            display_text = f"{icon} **{step_name}**"
        
        # 更新显示
        progress_dict["status_container"].markdown(display_text)


def render_sidebar() -> tuple:
    """渲染侧边栏配置"""
    with st.sidebar:
        st.header("⚙️ 配置")

        # 搜索参数
        search_type = st.selectbox(
            "搜索类型",
            ["hybrid", "text", "vector"],
            index=0,
            help="选择搜索方式：hybrid(混合)、text(文本)、vector(向量)",
        )

        top_k = st.slider(
            "检索文档数量",
            min_value=1,
            max_value=10,
            value=3,
            help="检索的相关文档数量",
        )

        st.divider()

        # 输出模式配置
        st.header("💬 输出模式")
        output_mode = st.radio(
            "选择输出方式",
            ["🚀 流式输出", "📄 完整输出"],
            index=0,
            help="流式输出：实时显示生成过程\n完整输出：等待完成后一次性显示",
        )

        show_intermediate = st.checkbox(
            "显示执行过程", value=True, help="显示各个组件的处理过程和中间结果"
        )

        show_document_content = st.checkbox(
            "显示文档详情", value=False, help="显示检索到的完整文档内容"
        )

        # 系统状态
        st.header("📊 系统状态")
        render_system_status()

        # 清空对话按钮
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return search_type, top_k, output_mode, show_intermediate, show_document_content


def render_system_status() -> None:
    """渲染系统状态"""
    if not st.session_state.pipeline_ready:
        with st.spinner("正在初始化系统..."):
            pipeline, ready = init_pipeline()
            st.session_state.pipeline = pipeline
            st.session_state.pipeline_ready = ready

    if st.session_state.pipeline_ready:
        st.success("✅ 系统就绪")
        if st.session_state.pipeline:
            components = st.session_state.pipeline.list_components()
            st.info(f"组件数量: {len(components)}")
    else:
        st.error("❌ 系统初始化失败")
        st.stop()


def format_document_sources(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """格式化文档来源信息"""
    sources = []
    for doc in documents[:3]:
        title = (
            doc.get("metadata", {}).get("filename")
            or doc.get("metadata", {}).get("source", "").split("/")[-1]
            or f"文档-{doc.get('id', 'N/A')}"
        )
        
        # 提取关键词信息
        keywords = []
        if 'highlights' in doc:
            highlights = doc['highlights']
            for field, terms_info in highlights.items():
                if isinstance(terms_info, dict) and 'relevant_terms' in terms_info:
                    keywords.extend(terms_info['relevant_terms'])
        
        content = doc.get("content", "")[:200]
        if '<mark>' in content:
            content = content.replace('<mark>', '**').replace('</mark>', '**')
            
        source_info = {
            "title": title,
            "content": content + "...",
            "score": doc.get("score", 0),
            "recall_source": doc.get('recall_source', 'unknown'),
            "keywords": list(set(keywords)) if keywords else []
        }
        sources.append(source_info)
    return sources


def render_sources(sources: List[Dict[str, Any]]) -> None:
    """渲染参考来源"""
    if sources:
        with st.expander("📚 参考来源"):
            for i, source in enumerate(sources, 1):
                recall_icon = "📝" if source.get('recall_source') == "text" else "🎯" if source.get('recall_source') == "vector" else "🔄"
                
                st.markdown(
                    f"**{i}. {recall_icon} {source['title']}** (相似度: {source['score']:.2f})"
                )
                
                # 显示关键词
                if source.get('keywords'):
                    st.markdown(f"🎯 **关键词**: {', '.join(source['keywords'])}")
                
                st.markdown(f"> {source['content']}")
                st.divider()


def render_chat_history() -> None:
    """渲染对话历史"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # 显示来源信息
            if "sources" in message and message["sources"]:
                render_sources(message["sources"])


def handle_streaming_response(
    prompt: str, top_k: int, search_type: str
) -> None:
    """处理流式响应"""
    # 在聊天界面中间创建进度指示器（独立于回复框）
    progress_placeholder = st.empty()
    
    # 创建助手回复框
    with st.chat_message("assistant"):
        # 创建占位符
        answer_placeholder = st.empty()
        sources_placeholder = st.empty()
        
        try:
            # 开始搜索流程
            progress_placeholder.markdown("🔄 正在执行: **文档检索** - 搜索相关文档...")
            
            # 执行搜索
            result_generator = search_with_pipeline_stream(prompt, top_k, search_type)
            
            final_result = None
            current_answer = ""
            
            for result in result_generator:
                # 根据实际的数据结构处理不同阶段
                if result.get("current_step") == "retrieval":
                    if result.get("step_state") == "running":
                        progress_placeholder.markdown("🔄 正在执行: **文档检索** - 搜索相关文档...")
                    elif result.get("step_state") == "completed":
                        progress_placeholder.markdown("✅ 已完成: **文档检索**")
                        
                elif result.get("current_step") == "rerank":
                    if result.get("step_state") == "running":
                        progress_placeholder.markdown("🔄 正在执行: **结果重排** - 重新排序文档...")
                    elif result.get("step_state") == "completed":
                        progress_placeholder.markdown("✅ 已完成: **结果重排**")
                        progress_placeholder.markdown("🔄 正在执行: **答案生成** - 生成回答...")
                        
                elif result.get("current_step") == "generation":
                    if result.get("step_state") == "running":
                        progress_placeholder.markdown("🔄 正在执行: **答案生成** - 生成回答...")
                        # 处理流式答案块
                        if result.get("answer_chunk"):
                            current_answer = result.get("answer_partial", current_answer + result["answer_chunk"])
                            answer_placeholder.markdown(current_answer + "▌")
                    elif result.get("step_state") == "completed":
                        progress_placeholder.markdown("✅ 已完成: **答案生成**")
                        
                elif result.get("current_step") == "completion":
                    if result.get("step_state") == "completed":
                        # 处理完成，清除进度指示器
                        progress_placeholder.empty()
                        
                # 检查是否为最终结果
                if result.get("is_final"):
                    final_result = result
                    current_answer = result.get("answer", current_answer)
                    answer_placeholder.markdown(current_answer)
                    # 清除进度指示器
                    progress_placeholder.empty()
                    break
                    
                # 处理错误
                if result.get("error"):
                    progress_placeholder.markdown(f"❌ 执行失败: {result.get('error', '未知错误')}")
                    st.error(f"处理过程中出现错误: {result.get('error', '未知错误')}")
                    return
            
            # 显示最终结果
            if final_result or current_answer:
                # 如果没有最终结果但有答案，创建一个基本的结果结构
                if not final_result:
                    final_result = {"answer": current_answer, "documents": []}
                
                # 格式化并显示来源
                sources = format_document_sources(final_result.get("documents", []))
                
                if sources:
                    with sources_placeholder.expander("📚 参考来源"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(
                                f"**{i}. {source['title']}** (相似度: {source['score']:.2f})"
                            )
                            st.markdown(f"> {source['content']}")
                            st.divider()
                
                # 添加到历史记录
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final_result.get("answer", current_answer),
                        "sources": sources,
                    }
                )
                
        except Exception as e:
            progress_placeholder.markdown(f"❌ 执行失败: {str(e)}")
            st.error(f"处理过程中出现错误: {str(e)}")
            logger.error(f"Streaming response error: {e}", exc_info=True)


def render_intermediate_results(
    result: Dict[str, Any], show_document_content: bool
) -> None:
    """渲染中间结果"""
    if "intermediate_results" not in result:
        return

    with st.expander("🔧 组件执行过程", expanded=True):
        for comp_name, comp_result in result["intermediate_results"].items():
            st.subheader(f"📦 {comp_name} ({comp_result['component_type']})")

            output = comp_result["output"]

            if comp_name == "es_retriever":
                render_retriever_results(output, show_document_content)
            elif comp_name == "zhipu_reranker":
                render_reranker_results(output, show_document_content)
            elif comp_name == "openai_generator":
                render_generator_results(output, show_document_content)

            # 显示原始数据
            with st.expander(f"🔍 原始数据 - {comp_name}", expanded=False):
                st.json(
                    {
                        "input": comp_result["input"],
                        "output": comp_result["output"],
                    }
                )

            st.divider()


def render_retriever_results(
    output: Dict[str, Any], show_document_content: bool
) -> None:
    """渲染检索器结果"""
    if "documents" in output:
        docs = output["documents"]
        st.success(f"✅ 检索到 {len(docs)} 个相关文档")

        if show_document_content:
            for i, doc in enumerate(docs[:5]):
                # 获取召回方式信息
                recall_source = doc.get('recall_source', 'unknown')
                recall_icon = "📝" if recall_source == "text" else "🎯" if recall_source == "vector" else "🔄"
                
                with st.expander(f"{recall_icon} 文档 {i+1} (相似度: {doc.get('score', 0):.3f}) - {recall_source}召回"):
                    st.write(f"**ID:** {doc.get('id', 'N/A')}")
                    
                    # 显示关键词高亮信息
                    if 'highlights' in doc:
                        st.write("**🎯 命中关键词:**")
                        highlights = doc['highlights']
                        for field, terms_info in highlights.items():
                            if isinstance(terms_info, dict) and 'relevant_terms' in terms_info:
                                relevant_terms = terms_info['relevant_terms']
                                if relevant_terms:
                                    st.write(f"  - **{field}**: {', '.join(relevant_terms)}")
                    
                    # 显示内容（保持原有的高亮标记）
                    content = doc.get('content', '')[:500]
                    if '<mark>' in content:
                        # 将HTML标记转换为Streamlit的markdown高亮
                        content = content.replace('<mark>', '**').replace('</mark>', '**')
                    st.write(f"**内容:** {content}...")
                    
                    if doc.get("metadata"):
                        st.write(f"**元数据:** {doc['metadata']}")
        else:
            for i, doc in enumerate(docs[:3]):
                recall_source = doc.get('recall_source', 'unknown')
                recall_icon = "📝" if recall_source == "text" else "🎯" if recall_source == "vector" else "🔄"
                
                # 显示简化的关键词信息
                keywords_info = ""
                if 'highlights' in doc:
                    highlights = doc['highlights']
                    all_terms = []
                    for field, terms_info in highlights.items():
                        if isinstance(terms_info, dict) and 'relevant_terms' in terms_info:
                            all_terms.extend(terms_info['relevant_terms'])
                    if all_terms:
                        keywords_info = f" [关键词: {', '.join(set(all_terms))}]"
                
                content = doc.get('content', '')[:100]
                if '<mark>' in content:
                    content = content.replace('<mark>', '**').replace('</mark>', '**')
                    
                st.write(
                    f"**{recall_icon} 文档 {i+1}** (得分: {doc.get('score', 0):.3f}) - {content}...{keywords_info}"
                )


def render_reranker_results(
    output: Dict[str, Any], show_document_content: bool
) -> None:
    """渲染重排器结果"""
    if "documents" in output:
        docs = output["documents"]
        st.success(f"✅ 重排后保留 {len(docs)} 个文档")

        if show_document_content:
            for i, doc in enumerate(docs):
                with st.expander(
                    f"📄 重排文档 {i+1} (重排得分: {doc.get('score', 0):.3f})"
                ):
                    st.write(f"**内容:** {doc.get('content', '')[:500]}...")
        else:
            for i, doc in enumerate(docs):
                st.write(
                    f"**重排文档 {i+1}** (得分: {doc.get('score', 0):.3f}) - {doc.get('content', '')[:100]}..."
                )


def render_generator_results(
    output: Dict[str, Any], show_document_content: bool
) -> None:
    """渲染生成器结果"""
    st.success("✅ 生成回答完成")
    if "context_used" in output:
        st.info(f"📚 使用了 {output['context_used']} 个文档作为上下文")

    if "documents" in output and show_document_content:
        with st.expander("📋 生成器输入的文档内容"):
            for i, doc in enumerate(output["documents"]):
                st.write(f"**输入文档 {i+1}:**")
                st.write(doc.get("content", "")[:300] + "...")
                st.divider()


def handle_non_streaming_response(
    prompt: str,
    top_k: int,
    search_type: str,
    show_intermediate: bool,
    show_document_content: bool,
) -> None:
    """处理非流式响应"""
    with st.spinner("🔍 正在搜索和生成回答..."):
        result = search_with_pipeline(prompt, top_k, search_type, show_intermediate)

    if result.get("answer"):
        # 显示中间结果
        if show_intermediate:
            render_intermediate_results(result, show_document_content)

        # 显示最终回答
        st.markdown(result["answer"])

        # 准备和显示来源信息
        sources = []
        if "documents" in result:
            sources = format_document_sources(result["documents"])

        render_sources(sources)

        # 添加到历史记录
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": sources,
            }
        )
    else:
        error_msg = f"抱歉，搜索时出现错误：{result.get('error', '未知错误')}"
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ==================== 主应用逻辑 ====================


def main():
    """主应用函数"""
    # 主界面标题
    st.title("🤖 RAG智能问答系统")
    st.markdown("基于Elasticsearch的知识检索与生成")

    # 渲染侧边栏配置
    search_type, top_k, output_mode, show_intermediate, show_document_content = (
        render_sidebar()
    )

    # 显示对话历史
    render_chat_history()

    # 聊天输入处理
    if prompt := st.chat_input("请输入你的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # 根据配置决定输出模式
        is_streaming = "流式" in output_mode

        if is_streaming:
            # 在用户提问和助手回复之间创建进度显示区域
            progress_placeholder = st.empty()
            
            with st.chat_message("assistant"):
                handle_streaming_response_with_progress(prompt, top_k, search_type, progress_placeholder)
        else:
            with st.chat_message("assistant"):
                handle_non_streaming_response(
                    prompt, top_k, search_type, show_intermediate, show_document_content
                )

        # 重新运行页面以显示新消息
        st.rerun()

    # 页面底部信息
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 12px;'>
            💡 提示：你可以询问关于性能分析、Linux工具、机器学习等技术问题
        </div>
        """,
        unsafe_allow_html=True,
    )


def handle_streaming_response_with_progress(
    prompt: str, top_k: int, search_type: str, progress_placeholder
) -> None:
    """处理流式响应，进度显示在独立位置"""
    # 创建占位符
    answer_placeholder = st.empty()
    sources_placeholder = st.empty()
    
    # CSS动画样式
    spinner_css = """
    <style>
    .spinner {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid #f3f3f3;
        border-top: 2px solid #3498db;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 8px;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .progress-text {
        color: #666;
        font-size: 14px;
        display: flex;
        align-items: center;
    }
    </style>
    """
    
    try:
        # 开始搜索流程
        progress_placeholder.markdown(
            spinner_css + '<div class="progress-text"><div class="spinner"></div>正在检索文档...</div>', 
            unsafe_allow_html=True
        )
        
        # 执行搜索
        result_generator = search_with_pipeline_stream(prompt, top_k, search_type)
        
        final_result = None
        current_answer = ""
        
        for result in result_generator:
            # 根据实际的数据结构处理不同阶段
            if result.get("current_step") == "retrieval":
                if result.get("step_state") == "running":
                    progress_placeholder.markdown(
                        spinner_css + '<div class="progress-text"><div class="spinner"></div>正在检索文档...</div>', 
                        unsafe_allow_html=True
                    )
                elif result.get("step_state") == "completed":
                    progress_placeholder.markdown(
                        spinner_css + '<div class="progress-text"><div class="spinner"></div>正在重排文档...</div>', 
                        unsafe_allow_html=True
                    )
                    
            elif result.get("current_step") == "rerank":
                if result.get("step_state") == "running":
                    progress_placeholder.markdown(
                        spinner_css + '<div class="progress-text"><div class="spinner"></div>正在重排文档...</div>', 
                        unsafe_allow_html=True
                    )
                elif result.get("step_state") == "completed":
                    progress_placeholder.markdown(
                        spinner_css + '<div class="progress-text"><div class="spinner"></div>正在生成回答...</div>', 
                        unsafe_allow_html=True
                    )
                    
            elif result.get("current_step") == "generation":
                if result.get("step_state") == "running":
                    progress_placeholder.markdown(
                        spinner_css + '<div class="progress-text"><div class="spinner"></div>正在生成回答...</div>', 
                        unsafe_allow_html=True
                    )
                    # 处理流式答案块
                    if result.get("answer_chunk"):
                        current_answer = result.get("answer_partial", current_answer + result["answer_chunk"])
                        answer_placeholder.markdown(current_answer + "▌")
                elif result.get("step_state") == "completed":
                    # 清除进度指示器
                    progress_placeholder.empty()
                    
            elif result.get("current_step") == "completion":
                if result.get("step_state") == "completed":
                    # 处理完成，清除进度指示器
                    progress_placeholder.empty()
                    
            # 检查是否为最终结果
            if result.get("is_final"):
                final_result = result
                current_answer = result.get("answer", current_answer)
                answer_placeholder.markdown(current_answer)
                # 清除进度指示器
                progress_placeholder.empty()
                break
                
            # 处理错误
            if result.get("error"):
                progress_placeholder.markdown(
                    f'<div style="color: #ff6b6b; font-size: 14px;">❌ 处理失败: {result.get("error", "未知错误")}</div>', 
                    unsafe_allow_html=True
                )
                st.error(f"处理过程中出现错误: {result.get('error', '未知错误')}")
                return
        
        # 在循环结束后处理最终结果（避免重复添加）
        if final_result or current_answer:
            # 如果没有最终结果但有答案，创建一个基本的结果结构
            if not final_result:
                final_result = {"answer": current_answer, "documents": []}
            
            # 格式化并显示来源
            sources = format_document_sources(final_result.get("documents", []))
            
            if sources:
                with sources_placeholder.expander("📚 参考来源"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(
                            f"**{i}. {source['title']}** (相似度: {source['score']:.2f})"
                        )
                        st.markdown(f"> {source['content']}")
                        st.divider()
            
            # 添加到历史记录（只添加一次）
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_result.get("answer", current_answer),
                    "sources": sources,
                }
            )
            
    except Exception as e:
        progress_placeholder.markdown(
            f'<div style="color: #ff6b6b; font-size: 14px;">❌ 处理失败: {str(e)}</div>', 
            unsafe_allow_html=True
        )
        st.error(f"处理过程中出现错误: {str(e)}")
        logger.error(f"Streaming response error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
