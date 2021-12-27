import datetime


def print_ts(text: str):
    """Prints text to stdout and includes timestamps at the beginning of each line.

    Args:
        text ([type]): Text to be incapsulated in the datetime wrapper.
    """
    print('[%s] %s' % (datetime.datetime.now(), text), flush=True)
