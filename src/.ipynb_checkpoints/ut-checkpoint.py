from enum import Enum
from termcolor import colored

class LogLevel(Enum):
    WARNING = 1
    INFO = 2
    ERROR = 3
    CRITICAL = 4
    DEBUG = 5
    PERFORMANCE = 6
    TARGET_NAME = 7
    

# Expose enum values at module level for convenient access
INFO = LogLevel.INFO
WARNING = LogLevel.WARNING
PERFORMANCE = LogLevel.PERFORMANCE
TARGET_NAME = LogLevel.TARGET_NAME
ERROR = LogLevel.ERROR
DEBUG = LogLevel.DEBUG
CRITICAL = LogLevel.CRITICAL

def log(level: LogLevel, s: str):
    match level:
        case LogLevel.INFO:
            print(colored("INFO", "green"), f": \t\t\t{s}")
        case LogLevel.PERFORMANCE:
            print(colored("PERFORMANCE", "green"), f": \t\t{s}")
        case LogLevel.WARNING:
            print(colored("WARNING", "yellow"), f": \t\t{s}")
        case LogLevel.TARGET_NAME:
            print(colored("\nTARGET NAME", "blue"), f": \t\t{s}")
        case LogLevel.DEBUG:
            print(colored("=" * (len(f": {s}") + 5 + 17), "red"), # + 5 for "DEBUG" string, + 16 for double tab
                  colored("\nDEBUG", "red"), f": \t\t{s}",
                  colored("\n" + "=" * (len(f": {s}") + 5 + 17), "red")
                 )
        case LogLevel.ERROR:
            print(colored("ERROR", "red"), f": \t\t{s}")
        case LogLevel.CRITICAL:
            print(colored("CRITICAL", "red"), f": \t\t{s}")
        case _:
            print(colored("ERROR", "red"), ": \t\tNO VALID LOGLEVEL FOUND, MAKE SURE TO NOT USE STRINGS AS 'level' PARAMETER")
