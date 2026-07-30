class Printer:
    """Prints a fixed "Hello World!" greeting to stdout.

    Example:
        >>> Printer().print_message()
        Hello World!
    """
    def __init__(self):
        """Create a Printer instance. Takes no arguments and holds no state."""
        pass

    def print_message(self):
        """Print "Hello World!" to the console."""
        print("Hello World!")

if __name__ == "__main__":
    iteration_counter = 0  # Added variable to count iterations
    # Override/use the class functionality
    printer = Printer()
    printer.print_message()
    iteration_counter += 1
    print('Total iterations:', iteration_counter)