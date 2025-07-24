import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline.factory import create_pipeline
from rag.components.loader.loader_utils import FileLoader
from utils.logger import setup_logging

def main():
    # 设置日志级别为DEBUG，这样可以看到debug级别的日志
    setup_logging(level="DEBUG")
    
    # 方法1：使用管道方式加载文件
    try:
        print("===== 使用管道方式加载文件 =====")
        pipeline = create_pipeline("file_loader_pipeline")
        print("成功创建管道")
        
        # 执行管道
        result = pipeline.run({})
        
        # 输出结果
        print(f"加载了 {len(result['documents'])} 个文档")
        print("\n文档内容预览:")
        for i, doc in enumerate(result['documents']):
            # 只显示前5个文档，每个文档显示前100个字符
            if i >= 5:
                break
            content = doc["content"]
            preview = content[:100] + "..." if len(content) > 100 else content
            print(f"\n文档 {i+1}:\n{preview}")
            
            # 显示元数据
            if "metadata" in doc and doc["metadata"]:
                print(f"元数据: {doc['metadata']}")
        
        # 方法2：直接使用FileLoader类
        print("\n\n===== 直接使用FileLoader类 =====")
        test_data_path = "/Users/caixiaomeng/Projects/Python/PerformanceRag/test_cases/test_data/"
        loader = FileLoader(path=test_data_path, file_types=[".txt", ".md"])
        documents = loader.load()
        
        print(f"直接加载了 {len(documents)} 个文档")
        print("\n文档内容预览:")
        for i, doc in enumerate(documents):
            if i >= 5:
                break
            content = doc["content"]
            preview = content[:100] + "..." if len(content) > 100 else content
            print(f"\n文档 {i+1}:\n{preview}")
            print(f"元数据: {doc['metadata']}")
            
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()