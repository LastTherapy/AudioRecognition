from openai import OpenAI
import os
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise Exception("OPENAI_API_KEY is not set in enviroment")
client = OpenAI(
  api_key=api_key
)

def improve_recognition(decrypted_audio: str):
  completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=[
      {"role": "user", "content": f"Это расшифровка голосового сообщения. Исправь грамматические, пунктационные, стилистические и смысловые ошибки так, чтобы расшифровка была наиболее точной. Не меняй порядок слов. Отвечай только расшифровку. Текст расшифроки: {decrypted_audio}" }
    ]
  )
  return completion.choices[0].message.content


if __name__ == "__main__":
  voice_message = "чтоб каждый понимал кто был у меня дома я нихуя не съел из холодильника за эти четыре дня вот пока я была потому что я спала и вот у меня остался тот остался оливье у меня осталась моя паста с курицей салат сырные тарелки карбонат салями у меня это все осталось и просовывал холодильник или глаз спать и я вот встала доброе утро"
  voice_message2 = "если ты хочешь жрать примеру тортик или бутерброды пожалуйста на кухне второго этажа"
  print(improve_recognition(voice_message2))
