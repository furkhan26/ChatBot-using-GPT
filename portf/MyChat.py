import unittest
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import transformers
print(transformers.__version__)
class ChatBot:
    def __init__(self, model_name="gpt2"):
        try:    
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        except Exception as e:
            raise ValueError(f"Error initializing model: {str(e)}")

    def generate_response(self, user_input):
        try:
            input_ids = self.tokenizer.encode(user_input, return_tensors="pt")
            output = self.model.generate(input_ids, max_length=150, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.5)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return response
        except Exception as e:
            raise ValueError(f"Error generating response: {str(e)}")

class TestChatBot(unittest.TestCase):
    def setUp(self):
        self.chatbot = ChatBot()

    def test_response_generation(self):
        user_input = "Hello, how are you?"
        response = self.chatbot.generate_response(user_input)
        self.assertIsInstance(response, str)

    def test_invalid_model_name(self):
        with self.assertRaises(ValueError):
            invalid_chatbot = ChatBot(model_name="invalid_model_name")

if __name__ == "__main__":
    ChatBot.generate_response()
    # unittest.main()
