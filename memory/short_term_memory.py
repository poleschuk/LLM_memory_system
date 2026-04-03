from collections import deque


class short_term_memory:
    def __init__(self, max_messages=20):
        self.short_term_memory = deque(maxlen=max_messages)

    def get(self):
        return list(self.short_term_memory)

    def push(self, message):
        self.short_term_memory.append(message)

    def pop(self):
        if not self.short_term_memory:
            return None
        return self.short_term_memory.popleft()
