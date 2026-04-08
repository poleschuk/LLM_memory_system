import uuid

from memory.embedding.compress import compressor

class embedding_database:
    def __init__(self, collections, model):
        self.collections = collections
        self.model = model

    def embed(self, text):
        return self.model.encode(text).tolist()

    def push_message(self, text, client, metadata: dict = None):
        text_full = "".join(msg["content"] for msg in text)

        comp = compressor(text_full, client)
        message_compresed = comp.compress()

        embeddings = self.embed(message_compresed)
        doc_id = str(uuid.uuid4())
        self.collections.add(
            documents=[message_compresed],
            embeddings=[embeddings],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def get(self):
        return self.collections

    def get_embeddings_documents(self, text):
        embeddings = self.embed(text)
        results = self.collections.query (
            query_embeddings=[embeddings],
            n_results=3
        )
        docs = results["documents"]
        if not docs or not docs[0]:
            return []

        return docs[0]