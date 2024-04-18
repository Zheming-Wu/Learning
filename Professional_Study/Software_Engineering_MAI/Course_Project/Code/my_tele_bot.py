# function part
import psutil
import time


def __proc_exist(process_name):
    pl = psutil.pids()
    for pid in pl:
        if psutil.Process(pid).name() == process_name:
            return pid


def __proc_terminate(pid):
    try:
        process = psutil.Process(pid)
        process.terminate()
        process.wait()  # Waiting for Process exit
        return True
    except psutil.NoSuchProcess:
        return False


def __proc_time(pid):
    for proc in psutil.process_iter(['pid', 'create_time']):
        if proc.info['pid'] == pid:
            pid_running_time = time.time() - proc.info['create_time']
            return pid_running_time
    pass


# telegram part
import telebot

with open("my_token.env", "r") as f:
    data = f.read()
    my_token = data.split('=')[1]

bot = telebot.TeleBot(my_token, parse_mode='MARKDOWN')  # You can set parse_mode by default. HTML or MARKDOWN


# Set up command bars
bot.delete_my_commands(scope=None, language_code=None)
bot.set_my_commands(
    commands=[
        telebot.types.BotCommand("/start", "Start Bot"),
        telebot.types.BotCommand("/help", "Help with Bot"),
        telebot.types.BotCommand("/check", "Check Process"),
        telebot.types.BotCommand("/terminate", "Terminate Process"),
    ],
)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    chat_id = message.chat.id
    bot.send_message(chat_id, text='Hello, what do you want to do on Process?')

    keyboard = telebot.types.ReplyKeyboardMarkup(row_width=2)
    button1 = telebot.types.KeyboardButton('/check')
    button2 = telebot.types.KeyboardButton('/terminate')
    keyboard.row(button1, button2)

    bot.send_message(chat_id, 'Check or Terminate?', reply_markup=keyboard)

    # bot.reply_to(message, "Hello, which Process do you want to check?")
    # bot.register_next_step_handler(message, check_process)


@bot.message_handler(commands=['check'])
def send_check_process(message):
    bot.reply_to(message, "Hello, which Process do you want to check?")
    bot.register_next_step_handler(message, check_process)


def check_process(message):
    process_name = message.text
    process_pid = __proc_exist(process_name)
    process_time = __proc_time(process_pid)
    if isinstance(process_pid, int):
        bot.reply_to(message, f'{process_name} is running for {process_time} seconds.')
    else:
        bot.reply_to(message, 'no such process...')


@bot.message_handler(commands=['terminate'])
def send_terminate_process(message):
    bot.reply_to(message, "Hello, which Process do you want to terminate?")
    bot.register_next_step_handler(message, terminate_process)


def terminate_process(message):
    process_name = message.text
    target_pid = __proc_exist(process_name)

    if target_pid:
        if __proc_terminate(target_pid):
            bot.reply_to(message, f"Successfully terminated Process {process_name} PID: {str(target_pid)}")
        else:
            bot.reply_to(message, f"Failed to terminate Process {process_name} PID: {str(target_pid)}")
    else:
        bot.reply_to(message, f"Target process {process_name} not found.")
    pass


bot.infinity_polling()
