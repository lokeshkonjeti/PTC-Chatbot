from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from pinecone import Pinecone, ServerlessSpec
from llama_index.core.settings import Settings
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.storage import StorageContext
import os

def pinecone_setup(chunk_size=512, t_dimensions=768, gemini_key="", api_key="", index_name="gemini-chatbot-3", documents=None):
    os.environ["GOOGLE_API_KEY"] = gemini_key
    llm = Gemini()
    embed_model = GeminiEmbedding(model_name="embedding-001")

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = chunk_size

    pc = Pinecone(api_key=api_key)
    index_names = [i.name for i in pc.list_indexes()]

    if index_name in index_names:
        print("📦 Index '{}' found. Entering query mode...".format(index_name))
        pinecone_index = pc.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        return VectorStoreIndex.from_vector_store(vector_store)

    elif documents is not None:
        print("🚧 Index '{}' not found. Creating and indexing now...".format(index_name))
        pc.create_index(
            name=index_name,
            dimension=t_dimensions,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        pinecone_index = pc.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    else:
        raise ValueError("Index not found and no documents provided to create it.")