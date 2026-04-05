import uuid

class embedding_database:
    def __init__(self, collections, model):
        self.collections = collections
        self.model = model

    def embed(self, text):
        return self.model.encode(text).tolist()

    def push_message(self, message, metadata: dict = None):
        docs = [message]
        embeddings = [self.embed(doc) for doc in docs]
        doc_id = str(uuid.uuid4())
        self.collections.add(
            documents=docs,
            embeddings=embeddings,
            metadatas=metadata,
            ids=[doc_id]
        )