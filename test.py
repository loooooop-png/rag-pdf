# 导入 create_pdf_rag 函数
from rag import load_pdf, get_result
import os

# 设置环境变量以配置 OpenAI API
os.environ["OPENAI_API_KEY"] = "none"  # 设置 OpenAI API 密钥（此处为示例，实际使用时请替换为有效密钥）
os.environ["OPENAI_API_BASE"] = "http://localhost:1337/v1"  # 设置 OpenAI API 基础 URL

# 主程序入口
if __name__ == "__main__":
    pdf_path = "OSCAT basic.pdf"  # 指定要处理的 PDF 文件路径
    query = "What is the function of V3_SMUL?"  # 指定要查询的问题

    try:
        # 调用 create_pdf_rag 函数处理 PDF 并获取结果
        retriever = load_pdf(pdf_path)
        result = get_result(query, retriever)
        
        # 打印答案
        print("Answer:", result["answer"])
        
        # 打印源文档信息
        print("\nSource Documents:")
        for doc in result["source_documents"]:
            # 打印每个文档的页码和内容的前 200 个字符
            print(f"Page {doc.metadata['page']}: {doc.page_content[:200]}...")
    except Exception as e:
        # 捕获并打印任何异常
        print(f"Error: {str(e)}")