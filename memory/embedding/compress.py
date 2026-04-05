from config.LLM_config import chat, MODEL_NAME

class compressor():
    def __init__(self, text, user, client):
        self.text = text
        self.user = user
        self.client = client

    def compress_with_llm(self):
        prompt = f"Summarize this memory in 1 short sentence:\n{self.text}"
        ai_reply = chat(self.client, MODEL_NAME, prompt)
        return ai_reply

    def compress_user(self):
        if "I" in self.text:
            return self.text.replace("I", self.user)
        return self.text

    def compress(self):
        user_text = self.compress_user()
        summ_text = self.compress_with_llm()
        return summ_text