class Printer:
    """
    A simple class to demonstrate printing functionality.
    """
    def __init__(self):
        """
        Initializes the Printer object.
        """
        pass

    def print_message(self):
        """
        Prints a greeting message to the console.
        """
        print("Hello World!")

if __name__ == "__main__":
    # Override/use the class functionality
    printer = Printer()
    printer.print_message()