"""
memory.py — Simple conversation memory for LexSearch
Stores last 5 Q&A pairs so users can ask follow-up questions.
Uses a plain Python list (no Redis needed for local dev).
"""

class ConversationMemory:
    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.history = []

    def add(self, question, answer):
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context(self):
        if not self.history:
            return ""
        lines = ["Previous conversation:"]
        for turn in self.history:
            lines.append(f"Q: {turn['question']}")
            lines.append(f"A: {turn['answer'][:200]}...")
        return "\n".join(lines)

    def clear(self):
        self.history = []

    def __len__(self):
        return len(self.history)
