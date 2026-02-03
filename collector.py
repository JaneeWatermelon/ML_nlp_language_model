import os
from telethon import TelegramClient
import asyncio
import vars
import dotenv

dotenv.load_dotenv()

API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
GROUP = os.getenv("GROUP")
OUTPUT_FILE = os.path.join(vars.DATASETS_ROOT, "dungeon_messages.txt")

async def main():
    async with TelegramClient("session", API_ID, API_HASH) as client:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            async for msg in client.iter_messages(GROUP, reverse=True):
                if msg.text:
                    author = msg.sender_id
                    date = msg.date.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{date}] {author}: {msg.text}\n\n")

asyncio.run(main())
