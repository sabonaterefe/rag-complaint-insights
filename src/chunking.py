from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_texts(texts, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return [chunk for text in texts for chunk in splitter.split_text(text)]
