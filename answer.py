import os
from docx import Document
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh import writing

# Создание схемы индекса
schema = Schema(title=TEXT(stored=True), content=TEXT, path=ID(stored=True))

# Создание индекса
if not os.path.exists("indexdir"):
    os.mkdir("indexdir")
ix = create_in("indexdir", schema)

# Индексирование файлов
def index_documents(directory):
    writer = ix.writer()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".docx"):
                path = os.path.join(root, file)
                doc = Document(path)
                content = "\n".join([para.text for para in doc.paragraphs])
                writer.add_document(title=file, content=content, path=path)
    writer.commit()

# Индексирование файлов в папке
index_documents("C:\Python\Dataset\docx_files")

# Поиск по индексу
def search_documents(query_str):
    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse(query_str)
        results = searcher.search(query)
        return results

# Пример использования
def predict_and_search(text):
    # Поиск документов
    results = search_documents(text)
    # Формирование ответа
    response = {
        "documents": []
    }
    for result in results:
        doc_info = {
            "title": result["title"],
            "path": result["path"],
            "content": result.highlights("content")
        }
        response["documents"].append(doc_info)
    return response

# Пример использования
result = predict_and_search("I can't connect to the internet")
print(result)
