import tkinter as tk
from tkinter import scrolledtext
from main import test_sparse_matrix
def test_sparse_matrix_gui():
    """
    Wrapper pentru a apela metoda test_sparse_matrix și a afișa rezultatele în interfața grafică.
    """
    output_text.delete(1.0, tk.END)  # Șterge textul existent din text box
    try:
        # Redirectează print-urile către text box
        class PrintRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget

            def write(self, text):
                self.text_widget.insert(tk.END, text)
                self.text_widget.see(tk.END)  # Scroll automat la final

            def flush(self):
                pass

        import sys
        sys.stdout = PrintRedirector(output_text)

        # Apelează metoda de test
        test_sparse_matrix()

        # Resetează stdout la valoarea originală
        sys.stdout = sys.__stdout__

    except Exception as e:
        output_text.insert(tk.END, f"Eroare: {e}\n")
        sys.stdout = sys.__stdout__  # Resetează stdout la valoarea originală


# Creează fereastra principală
root = tk.Tk()
root.title("Test Sparse Matrix")

# Creează un buton pentru a apela metoda de test
test_button = tk.Button(root, text="Rulează Test Sparse Matrix", command=test_sparse_matrix_gui)
test_button.pack(pady=10)

# Creează un text box scrollabil pentru a afișa rezultatele
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
output_text.pack(padx=10, pady=10)

# Rulează interfața grafică
root.mainloop()