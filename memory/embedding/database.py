import uuid

from memory.embedding.compress import compressor

class embedding_database:
    def __init__(self, collections, model):
        self.collections = collections
        self.model = model

    def embed(self, text):
        return self.model.encode(text).tolist()

    def push_message(self, text, client, metadata: dict = None):
        message = text["content"]
        user = text["role"]

        comp = compressor(message, user, client)
        message_compresed = comp.compress()

        docs = [message_compresed]
        embeddings = [self.embed(doc) for doc in docs]
        doc_id = str(uuid.uuid4())
        self.collections.add(
            documents=docs,
            embeddings=embeddings,
            metadatas=metadata,
            ids=[doc_id]
        )