import re
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict, Counter

# Убедимся, что необходимые ресурсы NLTK загружены
nltk.download('punkt')
nltk.download('wordnet')


def preprocess_text(text):
    # Лемматизация и очистка текста
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha()]
    return words


def build_markov_chain(words):
    # Строим Марковскую цепь первого и второго порядка
    chain = defaultdict(Counter)
    chain_first_order = defaultdict(Counter)

    for i in range(len(words) - 2):
        current_bigram = (words[i], words[i + 1])
        next_word = words[i + 2]
        chain[current_bigram][next_word] += 1

    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i + 1]
        chain_first_order[current_word][next_word] += 1

    return chain, chain_first_order


def generate_suggestion(chain, chain_first_order, input_words):
    # Попытка использовать биграммы, а затем униграммы
    if len(input_words) >= 2:
        last_bigram = (input_words[-2], input_words[-1])
        if last_bigram in chain:
            next_words = list(chain[last_bigram].keys())
            weights = list(chain[last_bigram].values())
            return random.choices(next_words, weights=weights, k=1)[0]

    if len(input_words) >= 1:
        last_word = input_words[-1]
        if last_word in chain_first_order:
            next_words = list(chain_first_order[last_word].keys())
            weights = list(chain_first_order[last_word].values())
            return random.choices(next_words, weights=weights, k=1)[0]

    return None


def main():
    try:
        # Чтение текста из файла
        with open('Атом.txt', 'r', encoding='utf-8') as file:
            text = file.read()

        words = preprocess_text(text)
        chain, chain_first_order = build_markov_chain(words)

        while True:
            user_input = input("Введите фразу (или 'exit' для выхода): ").strip().lower()
            if user_input == 'exit':
                break

            input_words = preprocess_text(user_input)

            suggestion = generate_suggestion(chain, chain_first_order, input_words)
            if suggestion:
                print("Подсказка:", suggestion)
            else:
                print("Нет подсказок для данной фразы.")
    except FileNotFoundError:
        print("Файл 'Атом.txt' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()
