import os
import requests
import vars
import json

SSAU_FILE = os.path.join(vars.DATASETS_ROOT, "ssau_news.json")
MAX_NEWS = 1000
limit = 50

url = "http://ssau.ru/api/news/"
stable_params = {
    "limit": limit,
    "lang": "ru",
}

if __name__ == "__main__":

    print(stable_params)

    news = []

    for i in range(0, MAX_NEWS, limit):
        params = stable_params | {
            "index": i,
        }
        response = requests.get(
            url=url,
            params=params
        )

        try:
            status = response.raise_for_status()
        except Exception as e:
            print(e)

        data = response.json()

        if data:
            news += data

        print(len(news))

    print(len(news))

    with open(SSAU_FILE, "w", encoding="UTF-8") as f:
        json.dump(news, f)