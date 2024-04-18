# function part
import psutil


def __proc_exist(process_name):
    pl = psutil.pids()
    for pid in pl:
        if psutil.Process(pid).name() == process_name:
            return pid


def __terminate_process(pid):
    try:
        process = psutil.Process(pid)
        process.terminate()
        process.wait()  # 等待进程退出
        return True
    except psutil.NoSuchProcess:
        return False


# telegram part
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton

my_token = ''

bot = telebot.TeleBot(my_token, parse_mode='MARKDOWN')  # You can set parse_mode by default. HTML or MARKDOWN


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Hello, which Process do you want to check?")
    bot.register_next_step_handler(message, check_process)


@bot.message_handler(commands=['check'])
def send_check_process(message):
    bot.reply_to(message, "Hello, which Process do you want to check?")
    bot.register_next_step_handler(message, check_process)


def check_process(message):
    process_name = message.text
    if isinstance(__proc_exist(process_name), int):
        bot.reply_to(message, f'{process_name} is running')
    else:
        bot.reply_to(message, 'no such process...')


@bot.message_handler(commands=['terminate'])
def send_terminate_process(message):
    bot.reply_to(message, "Hello, which Process do you want to terminate?")
    bot.register_next_step_handler(message, terminate_process)


def terminate_process(message):
    process_name = message.text
    target_pid = __proc_exist(process_name)
    target_process_name = process_name

    if target_pid:
        if __terminate_process(target_pid):
            bot.reply_to(message, f"Successfully terminated process PID: {str(target_pid)}")
        else:
            bot.reply_to(message, f"Failed to terminate process PID: {str(target_pid)}")
    else:
        bot.reply_to(message, f"Target process {target_process_name} not found.")
    pass


bot.infinity_polling()
