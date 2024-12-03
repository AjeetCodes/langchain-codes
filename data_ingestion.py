import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()

# os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
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
# print(text_documents)

## Pdf reader
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader('attention.pdf')
docs=loader.load()

# print(docs[0].page_content)

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)
# print(documents[:5])
# raise SystemExit("exit")
## Vector Embedding And Vector Store
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma

embedingModel = SentenceTransformer("all-MiniLM-L6-v2")

# Function to generate embeddings
class CustomEmbeddings:
    def embed_documents(self, texts):
        # return embedding_model.encode(texts, convert_to_numpy=True)
        return embedingModel.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, query):
        """Generate an embedding for a single query."""
        return embedingModel.encode(query, convert_to_numpy=True).tolist()
    
db = Chroma.from_documents(documents, CustomEmbeddings())
query = "Who are the authors of attention is all you need?"
retireved_results=db.similarity_search(query)
print(retireved_results[0].page_content)
