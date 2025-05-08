import os
import sys


class NoPrints:
    """
    Context manager that temporarily redirects standard output to null device,
    effectively silencing any print statements within its scope.

    This code contains content from Stack Overflow
    Source: https://stackoverflow.com/a/45669280

    Stack Overflow content is licensed under CC BY-SA 4.0
    (Creative Commons Attribution-ShareAlike 4.0 International License)
    https://creativecommons.org/licenses/by-sa/4.0/

    Code by Stack Overflow user: https://stackoverflow.com/users/2039471/alexander-c
    Modified: No substantial modifications from original code
    """

    def __init__(self):
        self.saved_stream = None

    def __enter__(self):
        # Store the current stdout stream
        self.saved_stream = sys.stdout
        # Redirect stdout to null device
        sys.stdout = open(os.devnull, 'w')
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # Close the null device file
        sys.stdout.close()
        # Restore the original stdout stream
        sys.stdout = self.saved_stream
