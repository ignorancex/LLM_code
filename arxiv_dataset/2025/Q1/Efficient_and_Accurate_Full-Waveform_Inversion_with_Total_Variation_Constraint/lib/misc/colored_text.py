from colorama import Fore, Style


class ColoredText:
    @staticmethod
    def green_up():
        return f"{Fore.GREEN}{Style.BRIGHT}↑{Fore.RESET}{Style.RESET_ALL}"

    @staticmethod
    def green_down():
        return f"{Fore.GREEN}{Style.BRIGHT}↓{Fore.RESET}{Style.RESET_ALL}"

    @staticmethod
    def red_up():
        return f"{Fore.RED}{Style.BRIGHT}↑{Fore.RESET}{Style.RESET_ALL}"

    @staticmethod
    def red_down():
        return f"{Fore.RED}{Style.BRIGHT}↓{Fore.RESET}{Style.RESET_ALL}"
