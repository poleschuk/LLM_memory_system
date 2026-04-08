from collections import deque

from memory.embedding.database import embedding_database

class short_term_memory:
    def __init__(self, max_messages=10):
        self.short_term_memory = deque(maxlen=max_messages)
        self.count = 0
        self.max_messages = max_messages

    def get(self):
        return list(self.short_term_memory)

    def push(self, text, embedding_database: embedding_database = None, client = None):
        if (len(self.short_term_memory) == self.max_messages and self.count == self.max_messages):
            if embedding_database:
                old_messages = list(self.short_term_memory)
                embedding_database.push_message(old_messages, client)
                self.count = 0
        self.short_term_memory.append(text)
        self.count = self.count + 1

    def pop(self):
        if not self.short_term_memory:
            return None
        return self.short_term_memory.popleft()
