from collections import deque

from memory.embedding.database import embedding_database

class short_term_memory:
    def __init__(self, max_messages=10):
        self.short_term_memory = deque(maxlen=max_messages)
        self.max_messages = max_messages

    def get(self):
        return list(self.short_term_memory)

    def push(self, message, embedding_database: embedding_database = None):
        if (len(self.short_term_memory) == self.max_messages):
            if embedding_database:
                old_message = self.short_term_memory[0]
                embedding_database.push_message(old_message["content"])
        self.short_term_memory.append(message)

    def pop(self):
        if not self.short_term_memory:
            return None
        return self.short_term_memory.popleft()
