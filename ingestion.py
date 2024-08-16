import chunk
from operator import index
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
# CharacterTextSplitter is a class that splits text into characters
# CharacterTextSplitter allows us to take long text and split it into chunk
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai import embeddings

load_dotenv()
if __name__ == "__main__":
    print("Interesting...")
    loader = TextLoader("/home/alcan/repo/intro-to-vector-dbs/mediumblog1.txt")
    document = loader.load()

    print("splitting text into characters") 
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    print("embedding chunks")

    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))

    print("done")