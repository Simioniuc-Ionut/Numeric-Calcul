import numpy as np
import tkinter as tk
from tkinter import messagebox, scrolledtext
from Tema2 import *
from Tema2_bonus import *

def read_input():
    """ Citește valorile introduse în interfață și le transformă în numpy arrays. """
    try:
        # Citire matrice A
        A = np.array([[float(entries_A[i][j].get()) for j in range(3)] for i in range(3)])

        # Citire vector dU (diagonala lui U)
        dU = np.array([float(entries_dU[i].get()) for i in range(3)])

        # Citire vector b (termenii liberi)
        b = np.array([float(entries_b[i].get()) for i in range(3)])

        # Afișăm datele citite în consolă
        print("\nMatricea A:\n", A)
        print("Diagonala U (dU):\n", dU)
        print("Vectorul b:\n", b)

        # Call the imported functions
        A_combined = lu_inplace_with_fixed_U_diag(A.copy(), dU)
        x_substitution= substitution(A_combined.copy(), dU, b)
        norm_result = compute_norm(A, x_substitution, b, 1e-14)

        # Extract L and U from A_combined
        L = np.tril(A_combined)
        U = np.triu(A_combined, k=1)
        n = A_init.shape[0]
        U_full = np.zeros_like(A_init)
        for i in range(n):
            U_full[i, i] = U_diag[i]
        U_full += U

        # Reconstruct A from L and U
        A_reconstructed = L @ U_full

        # Compute determinant
        det_A = compute_detA(A_combined, dU)

        # Display results in the text widget
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Combined matrix (in-place LU):\n{A_combined}\n\n")
        result_text.insert(tk.END, f"Matrix L:\n{L}\n\n")
        result_text.insert(tk.END, f"Matrix U:\n{U_full}\n\n")
        result_text.insert(tk.END, f"Product L * U:\n{A_reconstructed}\n\n")
        result_text.insert(tk.END, f"Original matrix A:\n{A}\n\n")
        result_text.insert(tk.END, "-"*50 + "\n")
        result_text.insert(tk.END, f"Det A = {det_A}\n\n")
        result_text.insert(tk.END, "-"*50 + "\n")
        result_text.insert(tk.END, f"Substitution method: {x_substitution}\n")
        result_text.insert(tk.END, "-"*50 + "\n")
        result_text.insert(tk.END, f"Norm of the solution: {norm_result}\n")

        # Bonus
        L_vec, U_vec = lu_decomposition_vector_storage(A, dU)
        y = direct_substitution(L_vec, b, 3)
        x = indirect_substitution(U_vec, y, 3)
        LU_product = reconstruct_LU(L_vec, U_vec, 3)
        result_text.insert(tk.END, "Bonus"+"-"*45 + "\n")
        result_text.insert(tk.END, f"Original matrix A:\n{A}\n\n")
        result_text.insert(tk.END, f"LU decomposition (reconstructed product LU):\n{LU_product}\n\n")
        result_text.insert(tk.END, f"Solution of the system Ax=b (x):\n{x}\n\n")
        # Afișăm un mesaj de succes
        messagebox.showinfo("Succes", "Datele au fost preluate cu succes!")

    except ValueError:
        messagebox.showerror("Eroare", "Introduceți doar numere valide!")

# Creare fereastră principală
root = tk.Tk()
root.title("Tema nr. 2")  # Titlu fereastră

# Centrare fereastră pe ecran
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = 900
window_height = 650
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

# Creăm un cadru principal pentru centrare
frame = tk.Frame(root)
frame.pack(expand=True)

# Titlu mare centrat
title_label = tk.Label(frame, text="Tema nr. 2", font=("Arial", 16, "bold"))
title_label.grid(row=0, column=0, columnspan=5, pady=10)

# Eticheta pentru matricea A
tk.Label(frame, text="Matricea A (3x3)", font=("Arial", 12)).grid(row=1, column=0, columnspan=3)

# Creăm casetele pentru matricea A
entries_A = [[tk.Entry(frame, width=5, font=("Arial", 12), justify="center") for j in range(3)] for i in range(3)]
for i in range(3):
    for j in range(3):
        entries_A[i][j].grid(row=i+2, column=j, padx=5, pady=5)

# Eticheta pentru vectorul dU
tk.Label(frame, text="dU (diagonala U)", font=("Arial", 12)).grid(row=1, column=3, padx=10)
entries_dU = [tk.Entry(frame, width=5, font=("Arial", 12), justify="center") for i in range(3)]
for i in range(3):
    entries_dU[i].grid(row=i+2, column=3, padx=10, pady=5)

# Eticheta pentru vectorul b
tk.Label(frame, text="b (termeni liberi)", font=("Arial", 12)).grid(row=1, column=4, padx=10)
entries_b = [tk.Entry(frame, width=5, font=("Arial", 12), justify="center") for i in range(3)]
for i in range(3):
    entries_b[i].grid(row=i+2, column=4, padx=10, pady=5)

# Butonul Enter centrat
btn_enter = tk.Button(frame, text="Enter", font=("Arial", 12, "bold"), command=read_input, bg="lightblue")
btn_enter.grid(row=5, column=0, columnspan=5, pady=15)

# Creăm un widget text scrollabil pentru afișarea rezultatelor
result_text = scrolledtext.ScrolledText(frame, width=80, height=20, font=("Arial", 12))
result_text.grid(row=6, column=0, columnspan=5, pady=10)

# Rulăm fereastra
root.mainloop()