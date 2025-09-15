from llama_index.readers.web import BeautifulSoupWebReader

def load_documents(urls):
    reader = BeautifulSoupWebReader()
    return reader.load_data(urls=urls)