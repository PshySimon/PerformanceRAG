from utils.rag.query import RewriteExpansion
from utils.config import config

def main():
    expander = RewriteExpansion()
    query = "苹果手机电池不耐用怎么办？"
    rewritten = expander.transform(query)
    print(f"原始Query: {query}")
    print(f"改写后: {rewritten}")

if __name__ == "__main__":
    main() 