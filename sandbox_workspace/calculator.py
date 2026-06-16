import tkinter as tk
from tkinter import messagebox

class CalculatorApp:
    def __init__(self, master):
        self.master = master
        master.title("Calculator")
        master.geometry("300x400")

        self.current_expression = ""

        # Entry widget for the display
        self.display = tk.Entry(master, font=('Arial', 16), bd=5, relief=tk.SUNKEN, justify='right')
        self.display.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        # Define buttons
        buttons = [
            ('7', 1, 0), ('8', 1, 1), ('9', 1, 2), ('/', 1, 3),
            ('4', 2, 0), ('5', 2, 1), ('6', 2, 2), ('*', 2, 3),
            ('1', 3, 0), ('2', 3, 1), ('3', 3, 2), ('-', 3, 3),
            ('0', 4, 0), ('.', 4, 1), ('=', 4, 2), ('+', 4, 3),
            ('C', 5, 0)
        ]

        # Create and place buttons
        for (text, row, col) in buttons:
            if text == '=':
                btn = tk.Button(master, text=text, padx=20, pady=20, font=('Arial', 12), command=self.calculate)
            elif text == 'C':
                btn = tk.Button(master, text=text, padx=20, pady=20, font=('Arial', 12), command=self.clear)
            else:
                btn = tk.Button(master, text=text, padx=20, pady=20, font=('Arial', 12), command=lambda t=text: self.append_to_expression(t))
            
            btn.grid(row=row, column=col, sticky="nsew")

        # Configure grid weights for resizing
        for i in range(6):
            master.grid_rowconfigure(i, weight=1)
        for i in range(4):
            master.grid_columnconfigure(i, weight=1)

    def append_to_expression(self, value):
        self.current_expression += str(value)
        self.update_display()

    def clear(self):
        self.current_expression = ""
        self.update_display()

    def update_display(self):
        self.display.delete(0, tk.END)
        self.display.insert(0, self.current_expression)

    def calculate(self):
        try:
            result = str(eval(self.current_expression))
            self.current_expression = result
            self.update_display()
        except Exception as e:
            self.current_expression = "Error"
            self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = CalculatorApp(root)
    root.mainloop()