import os
from dotenv import load_dotenv
from dataservice import DataService
from intentservice import IntentService
from responseservice import ResponseService
import logging

# Настройка логирования
logging.basicConfig(
    filename='app.log',  # Логи будут записываться в app.log
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Загрузка переменных окружения
load_dotenv()

# Инициализация сервисов
data_service = DataService()
intent_service = IntentService()
response_service = ResponseService()

# Лимит вопросов за сессию
MAX_QUESTIONS = 10

# Функция для обработки одного вопроса
def process_question(question: str):
    try:
        logging.info("Начинаем обработку вопроса...")
        # Шаг 1: Получение намерения
        intents = intent_service.get_intent(question)
        logging.info(f"Намерение: {intents}")

        # Шаг 2: Поиск фактов в Redis
        facts = data_service.search_redis(intents)
        logging.info(f"Найденные факты: {facts}")

        # Шаг 3: Генерация ответа
        answer, usage = response_service.generate_response_with_usage(facts, question)
        logging.info(f"Ответ: {answer}")
        logging.info(f"Статистика по токенам: {usage}")

        # Показ ответа и статистики пользователю
        print(f"\nОтвет: {answer}")
        print(f"Использование токенов: "
              f"Промпт: {usage['prompt_tokens']}, "
              f"Ответ: {usage['completion_tokens']}, "
              f"Всего: {usage['total_tokens']}\n")
    except Exception as e:
        logging.error(f"Ошибка обработки вопроса: {e}")
        print(f"Ошибка: {e}")

# Основной цикл
if __name__ == "__main__":
    try:
        # Загрузка данных в Redis
        logging.info("Начинаем загрузку данных из PDF...")
        pdf_path = "ExplorersGuide.pdf"
        data_service.drop_redis_data()  # Удаляем существующие данные
        data = data_service.pdf_to_embeddings(pdf_path)  # Читаем PDF и создаём эмбеддинги
        data_service.load_data_to_redis(data)  # Загружаем данные в Redis
        logging.info("Данные успешно загружены в Redis.")

        # Цикл вопросов и ответов
        question_count = 0
        while question_count < MAX_QUESTIONS:
            question = input("Введите ваш вопрос (или 'выход' для завершения, 'помощь' для справки): ").strip()
            if question.lower() == "выход":
                print("Завершаем работу. До свидания!")
                break
            elif question.lower() == "помощь":
                print("\nВведите вопрос на русском языке, чтобы получить ответ. Команды:\n"
                      "- 'выход': завершить работу\n"
                      "- 'помощь': показать справку\n")
                continue

            # Проверка длины вопроса
            if len(question.split()) > 100:
                print("Ваш вопрос слишком длинный. Пожалуйста, уменьшите количество слов.")
                continue

            # Обработка вопроса
            process_question(question)
            question_count += 1

        print("Вы достигли лимита вопросов за сессию. Перезапустите программу для новой сессии.")
    except Exception as e:
        logging.error(f"Ошибка в основной программе: {e}")
        print(f"Критическая ошибка: {e}")
