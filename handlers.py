from aiogram import F, Router, Bot
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.fsm.context import FSMContext

import utils
from PIL import Image
import text

router = Router()

@router.message(Command("start"))
async def start_handler(msg: Message, state: FSMContext):
    await msg.answer(text.greet.format(name=msg.from_user.full_name))

@router.message(F.photo)
async def req4___(msg: Message, state: FSMContext, bot: Bot):
    content = await bot.download(msg.photo[-1])
    await msg.answer(utils.predict_product(Image.open(content)))

