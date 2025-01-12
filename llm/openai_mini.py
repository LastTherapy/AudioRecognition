from openai import OpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise Exception("OPENAI_API_KEY is not set in цenvironment")
client = OpenAI(
    api_key=api_key
)


def improve_recognition(decrypted_audio: str):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        temperature=0.2,  # Чем ниже, тем менее «творческой» будет модель
        top_p=0.9,  # Часто используют в районе 0.8–0.9
        presence_penalty=0.0,  # При коррекции нежелательно «штрафовать» за новые/повторяющиеся слова
        frequency_penalty=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты — помощник, который корректирует русскоязычные транскрипции. "
                    "Твоя задача — исправлять орфографические, пунктуационные, стилистические и смысловые ошибки, "
                    "при этом не меняя порядок слов и не выкидывая их. Если встречается сленг или транскрипция "
                    "английских слов, замени их на корректные английские слова, понятные в контексте варианты. "
                    "Старайся не удалять ничего и не добавлять ничего сверх необходимости. "
                    "Добавляй абзацы только там, где это логически обоснованно."
                )
            },
            {
                "role": "user",
                "content": (
                    "Это расшифровка голосового сообщения. "
                    "Не выбрасывай слова, даже если не понимаешь их значения. "
                    "Эта расшифровка будет отправлена как сообщение в Telegram. Поэтому расставь абзацы если текст будет длинным."
                    "Отвечай только расшифровку.\n\n"
                    f"Текст голосового: {decrypted_audio}"
                )
            }
        ]
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    voice_message = "чтоб каждый понимал кто был у меня дома я нихуя не съел из холодильника за эти четыре дня вот пока я была потому что я спала и вот у меня остался тот остался оливье у меня осталась моя паста с курицей салат сырные тарелки карбонат салями у меня это все осталось и просовывал холодильник или глаз спать и я вот встала доброе утро"
    voice_message2 = "если ты хочешь жрать примеру тортик или бутерброды пожалуйста на кухне второго этажа"
    print(improve_recognition(voice_message))