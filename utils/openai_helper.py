import openai
import os

class OpenAIHelper:
    """
    Helper class for openai queries
    """
    def __init__(self):
        # Set up your OpenAI API credentials
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_prompt = "You are a helpful assistant who will help students learn technology"
        self.conversation = [
            {"role": "system", "content": self.base_prompt},
        ]     

    def generate_assistant_response(self, message):
        """
        Generate ai assistant response
        """
        self.conversation.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=self.conversation
        )
        if response.ok:
            bot_response = response["choices"][0]["message"]["content"].strip()
            self.conversation.append({"role": "assistant", "content": bot_response})
        else:
            bot_response = "Sorry, I didn't understand that."
        return bot_response, self.conversation
    
    def reset_conversation(self):
        """
        Reset conversation
        """
        self.conversation = [
            {"role": "system", "content": self.base_prompt},
        ]
    
    def restore_conversation(self, conversation):
        """
        Restore conversation
        """
        self.conversation = conversation
    

