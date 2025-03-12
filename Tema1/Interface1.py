import tkinter as tk
from tkinter import scrolledtext
import subprocess

def run_exercise(script, exercise):
    result_text.delete(1.0, tk.END)
    process = subprocess.Popen(['python', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        result_text.insert(tk.END, stdout.decode('utf-8'))
    else:
        result_text.insert(tk.END, stderr.decode('utf-8'))

def run_ex1_ex2():
    run_exercise('Tema1_ex1_2.py', 'ex1_2')

def run_ex3():
    run_exercise('Tema1_ex3.py', 'ex3')

root = tk.Tk()
root.title("Teme nr.1")

# Center the window on the screen
window_width = 800
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)
root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

frame = tk.Frame(root)
frame.pack(pady=20)

btn_ex1 = tk.Button(frame, text="Ex1 and Ex2", font=("Arial", 12), command=run_ex1_ex2)
btn_ex1.grid(row=0, column=0, padx=10)

btn_ex3 = tk.Button(frame, text="Ex3", font=("Arial", 12), command=run_ex3)
btn_ex3.grid(row=0, column=2, padx=10)

result_text = scrolledtext.ScrolledText(root, width=80, height=20, font=("Arial", 12))
result_text.pack(pady=10)

root.mainloop()