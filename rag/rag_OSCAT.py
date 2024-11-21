# 导入所需的库
import os
from langchain_community.document_loaders import PyPDFLoader  # 用于加载 PDF 文档
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 用于将文本分割成块
from langchain_community.embeddings import OpenAIEmbeddings  # 用于生成 OpenAI 嵌入
from langchain_community.vectorstores import Chroma  # 用于创建向量存储
from langchain_community.chat_models import ChatOpenAI  # 用于与 OpenAI 聊天模型交互
from langchain.chains import RetrievalQA  # 用于创建问答链
from dotenv import load_dotenv  # 用于加载环境变量
from langchain_huggingface import HuggingFaceEmbeddings  # 用于 Hugging Face 嵌入

# 加载环境变量
load_dotenv()

def load_pdf(pdf_path):
    # 加载 PDF 文件
    loader = PyPDFLoader(pdf_path)  # 创建 PDF 加载器
    pages = loader.load()  # 加载 PDF 中的所有页面
    
    # 将文本分割成块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每个块的最大字符数
        chunk_overlap=200,  # 块之间的重叠字符数
        length_function=len  # 用于计算文本长度的函数
    )
    splits = text_splitter.split_documents(pages)  # 分割文档

    # 创建嵌入和向量存储
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')  # 使用 Hugging Face 嵌入模型
    vectorstore = Chroma.from_documents(
        documents=splits,  # 使用分割后的文档
        embedding=embedding_model  # 使用嵌入模型
    )

    # 创建检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # 使用相似性搜索
        search_kwargs={"k": 3}  # 返回最相似的 3 个文档
    )
    return retriever

def get_result(query, retriever):
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.7)  # 初始化聊天模型
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,  # 使用的语言模型
        chain_type="stuff",  # 链的类型
        retriever=retriever,  # 使用的检索器
        return_source_documents=True  # 返回源文档
    )
    # 获取响应
    result = qa_chain({"query": query})  # 传入查询并获取结果
    return {
        "answer": result["result"],  # 返回答案
        "source_documents": result["source_documents"]  # 返回源文档
    }


if __name__ == "__main__":
    pdf_path = r"..\OSCAT basic.pdf"  # 指定要处理的 PDF 文件路径
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