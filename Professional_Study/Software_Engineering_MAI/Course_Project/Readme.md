# Children computer game time control program 

The software will serve as a mobile application for parents to use on their smartphones to monitor and control their children's computer use behavior.

Author1: Wu Zheming 

Author2: Liang Yifan 

## 1. Installation 

### 1.1 Download code

You can use the 'git pull' command or 'web download' to download this code to the local.

### 1.2 Obtain TeleBot token 

Get the TeleBot token and put the token in the my_token.env file in the following form. You can [get the TeleBot token at Telegram @BotFather](https://core.telegram.org/bots#botfather). ('wwwzzzmmm' is your own token)

```
BOT_TOKEN=wwwzzzmmm
```

## 2. Quick start

The UI of this software is shown in the figure, which includes three parts: "Main functions", "Function instruction set" and "Information interaction area". 

![UI.png](https://github.com/Zheming-Wu/Learning/blob/main/Professional_Study/Software_Engineering_MAI/Course_Project/Figures/UI.png)

Enter '/start' on the Telegram chat interface to launch our bot. Enter '/help' on the Telegram chat interface to get some basic instructions.

![start.png](https://github.com/Zheming-Wu/Learning/blob/main/Professional_Study/Software_Engineering_MAI/Course_Project/Figures/start.png)

Click the '/check' button to send the name of the process to be checked, and the chat box will return the running status of the process. 

![check.png](https://github.com/Zheming-Wu/Learning/blob/main/Professional_Study/Software_Engineering_MAI/Course_Project/Figures/check.png)

Click '/terminate' and send the name of the process to be interrupted. The program interrupts the specified process and returns the result through the chat box.

![terminate.png](https://github.com/Zheming-Wu/Learning/blob/main/Professional_Study/Software_Engineering_MAI/Course_Project/Figures/terminate.png)

Click the '/monitor' button to set the maximum running time of the blacklist process (black_list.txt). When the blacklist process exceeds the specified maximum running time, the program will interrupt the specified process and return the interrupt result through the chat box.

![monitor.png](https://github.com/Zheming-Wu/Learning/blob/main/Professional_Study/Software_Engineering_MAI/Course_Project/Figures/monitor.png)

Click the '/report' button, and the chat box returns the running time report of the blacklist process.

![report.png](https://github.com/Zheming-Wu/Learning/blob/main/Professional_Study/Software_Engineering_MAI/Course_Project/Figures/report.png)

Enter '/add_blacklist' on the Telegram chat interface, the process can be added or deleted from the blacklist (black_list.txt). 

![add_blacklist.png](https://github.com/Zheming-Wu/Learning/blob/main/Professional_Study/Software_Engineering_MAI/Course_Project/Figures/add_blacklist.png)

Commands and corresponding functions are shown in the Table. 

|Command|Function|
| ---- | ---- |
| /start | Start Bot |
| /check | Help with Bot |
| /terminate | Terminate Process |
| /monitor | Auto Terminate Process |
| /report | Export Report |
| /add_blacklist | Add/delete Blacklist Process |

## 3. Add functions 

If you want to add new functionality to an existing code framework, follow the template requirements below. 

Take the implementation of the '/check' function for example. 

First, create an instruction response function that responds to the user's '/check' instruction and returns an input boot statement in the chat box. Gets the information re-entered by the user and sends it to the main functional function.

```python
@bot.message_handler(commands=['check'])
def send_check_process(message):
    bot.reply_to(message, "Hello, which Process do you want to check?")
    bot.register_next_step_handler(message, check_process)
```

Then, in the main functional function, we implement the corresponding function based on the user input information. 

```python
def check_process(message):
    process_name = message.text
    process_pid = __proc_exist(process_name)
    process_time = __proc_time(process_pid)
    if isinstance(process_pid, int):
        bot.reply_to(message, f'{process_name} is running for {process_time} seconds.')
    else:
        bot.reply_to(message, 'no such process...')
```

Finally, we run the code to test our new functionality via TeleBot. 

More information on TeleBot development can be found at [Telegram Bot API](https://core.telegram.org/bots/api).