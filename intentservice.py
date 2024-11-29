import openai
from dotenv import load_dotenv
import os

# Загрузка переменных из .env
load_dotenv()

# Инициализация OpenAI API ключа из .env
openai.api_key = os.getenv("OPENAI_API_KEY")

class IntentService:
    def __init__(self):
        pass

    def get_intent(self, user_question: str):
        # Вызов метода ChatCompletion для модели GPT-3.5-turbo
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f'Extract the keywords from the following question: {user_question}. '
                        'Do not answer anything else, only the keywords.'
                    )
                }
            ]
        )

        # Возврат ответа
        return response['choices'][0]['message']['content']


# Пример использования
if __name__ == "__main__":
    intent_service = IntentService()
    user_question = "What is the capital of France?"
    keywords = intent_service.get_intent(user_question)
    print("Extracted keywords:", keywords)
