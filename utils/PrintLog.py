from datetime import datetime

def d(message):
    print("DEBUG: {} - {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), message))


def w(message):
    print("WARNNING: {} - {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), message))


def e(message):
    print("ERROR: {} - {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), message))


def v(message):
    print("VERBOSE: {} - {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), message))


def i(message):
    print("INFO: {} - {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), message))

