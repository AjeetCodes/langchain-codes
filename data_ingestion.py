import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()

# text documents loader
loader = TextLoader("speech.txt")
text_documents = loader.load()
# print(text_documents)

# web based loader
from langchain_community.document_loaders import WebBaseLoader
import bs4

## load,chunk and index the content of the html page

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-title", "post-content", "post-header")
        )
    )
)

text_documents = loader.load()
print(text_documents)
# os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
