from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters.callback_data import CallbackData

class Feedback(CallbackData, prefix='fb'): 
    trace_id: str
    val: int 

def get_feedback_keyboard(trace_id: str):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
            InlineKeyboardButton(text="👍", 
                                 callback_data=Feedback(trace_id=trace_id, val=1).pack()
                                 ),
            InlineKeyboardButton(text="👎", 
                                 callback_data=Feedback(trace_id=trace_id, val=0).pack()
                                 ),
            ]                
        ]
    )
    return keyboard