from typing import List, Tuple, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


class Retriever:
    """
    Clase para implementar la recuperación de documentos basados en embeddings.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", data_path: str = "../../data/urls.txt"):
        """
        Inicializa el retriever con un modelo de embeddings y una ruta para los datos precomputados.

        Args:
            model_name (str): Nombre del modelo de FastEmbed.
            data_path (str): Ruta donde se almacenan los datos.
        """
        self.model = FastEmbedEmbeddings(model_name=model_name)
        self.data_path = data_path
        self.index = None
        self.documents = None

    def load_urls(self):
        """
        Carga un archivo de texto que contiene URLs en cada línea y devuelve una lista de URLs.

        Args:
            file_path (str): Ruta al archivo de texto.

        Returns:
            List[str]: Lista de URLs.
        """
        with open(self.data_path, "r", encoding="utf-8") as file:
            urls = [line.strip() for line in file if line.strip()]
        return urls

    def load_data(self):
        """
        Carga embeddings y documentos asociados desde el disco.
        """
        docs = [WebBaseLoader(url).load() for url in self.urls]
        docs_list = [item for sublist in docs for item in sublist]
        print(f"len of documents :{len(docs_list)}")
        return docs_list

    def create_embeddings(self, documents: List[str], index_path: str, docs_path: str):
        """
        Genera embeddings para los documentos y crea un índice FAISS.

        Args:
            documents (List[str]): Lista de documentos para indexar.
            index_path (str): Ruta para guardar el índice FAISS.
            docs_path (str): Ruta para guardar los documentos.
        """
        """
        # Generar embeddings
        embeddings = self.model.encode(documents, convert_to_numpy=True)

        # Crear y entrenar el índice FAISS
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # Guardar índice y documentos
        faiss.write_index(self.index, index_path)
        with open(docs_path, "w", encoding="utf-8") as f:
            f.writelines(f"{doc}\n" for doc in documents)

        self.documents = documents"""
        print("hello")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Recupera los documentos más relevantes para una consulta dada.

        Args:
            query (str): La consulta en texto libre.
            top_k (int): Número de documentos relevantes a devolver.

        Returns:
            List[Tuple[str, float]]: Lista de documentos relevantes y sus puntuaciones de similitud.
        """
        # Codificar la consulta
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Buscar en el índice FAISS
        distances, indices = self.index.search(query_embedding, top_k)

        # Mapear resultados a documentos
        results = [
            (self.documents[idx], float(dist)) for idx, dist in zip(indices[0], distances[0]) if idx != -1
        ]

        return results



if __name__ == "__main__":
    retriever = Retriever()

    print(retriever.load_urls())