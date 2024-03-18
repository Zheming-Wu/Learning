# Author: Wu Zheming


def fun(language, number):
    """
    :param language: English, Chinese, Russian
    :param number: Number of times to print results
    :return: Write “Hello” or “Nihao” or “Privet” on the screen number times
    """

    number = int(number)

    if language == 'English':
        text_to_print = 'Hello'
    elif language == 'Chinese':
        text_to_print = 'Nihao'
    elif language == 'Russian':
        text_to_print = 'Privet'
    else:
        text_to_print = None
        pass

    if text_to_print:
        for i in range(number):
            print(text_to_print)
    else:
        print('The language is not in our function!')


if __name__ == "__main__":
    fun(language='Chinese', number=2)
    fun(language='Russian', number=3)
    fun(language='English', number=1)
