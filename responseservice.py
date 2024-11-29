import openai

class ResponseService:
    def __init__(self):
        pass

    def generate_response_with_usage(self, facts, user_question):
        try:
            # Генерация ответа через OpenAI ChatCompletion
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"На основе следующих фактов ответьте на вопрос пользователя.\n"
                                   f"ФАКТЫ: {facts}\n"
                                   f"ВОПРОС: {user_question}"
                    }
                ]
            )

            # Извлечение ответа и статистики токенов
            summary = response.choices[0].message.content
            usage = response.usage  # Получаем статистику по токенам
            return summary, usage
        except openai.error.OpenAIError as e:
            logging.error(f"Ошибка OpenAI API: {e}")
            return "Ошибка при взаимодействии с OpenAI API.", None
