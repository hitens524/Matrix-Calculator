import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog # Import simpledialog for scalar/exponent input

class MatrixCalculatorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Matrix Calculator")
        master.geometry("800x700") 
        master.configure(bg="#f0f0f0") 

        # --- Styling ---
        self.button_font = ("Arial", 10, "bold")
        self.label_font = ("Arial", 12, "bold")
        self.text_font = ("Courier New", 10)
        self.button_bg = "#4CAF50" # Green
        self.button_fg = "white"
        self.entry_bg = "#ffffff"
        self.frame_bg = "#e0e0e0" 
        
        # --- Main Frames ---
        self.input_frame = tk.LabelFrame(master, text="Matrix Input", font=self.label_font, bg=self.frame_bg, fg="#333", bd=2, relief="groove")
        self.input_frame.pack(pady=10, padx=10, fill="x", expand=False)

        self.output_frame = tk.LabelFrame(master, text="Matrix Output", font=self.label_font, bg=self.frame_bg, fg="#333", bd=2, relief="groove")
        self.output_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.button_frame = tk.Frame(master, bg="#d0d0d0", bd=2, relief="raised")
        self.button_frame.pack(pady=10, padx=10, fill="x", expand=False)

        # --- Input Section ---
        input_info_label = tk.Label(self.input_frame, text="Enter matrix values separated by spaces for columns and newlines for rows. Use 'MATRIX2' to separate two matrices for operations like Addition/Multiplication. E.g., '1 2\\n3 4\\nMATRIX2\\n5 6\\n7 8'", font=("Arial", 9, "italic"), bg=self.frame_bg, fg="#555")
        input_info_label.pack(pady=(0,5), padx=5)

        self.input_text = scrolledtext.ScrolledText(self.input_frame, wrap=tk.WORD, width=60, height=8, font=self.text_font, bg=self.entry_bg, fg="#333", bd=1, relief="solid")
        self.input_text.pack(pady=5, padx=5, fill="both", expand=True)
        
        # --- Output Section ---
        self.output_text = scrolledtext.ScrolledText(self.output_frame, wrap=tk.WORD, width=60, height=15, font=self.text_font, bg=self.entry_bg, fg="#333", bd=1, relief="solid", state=tk.DISABLED)
        self.output_text.pack(pady=5, padx=5, fill="both", expand=True)
        
        # --- Buttons Section ---
        button_row1 = tk.Frame(self.button_frame, bg=self.button_frame['bg'])
        button_row1.pack(pady=5, fill="x")
        self.create_button(button_row1, "Input & Display", self.display_input_matrix)
        self.create_button(button_row1, "Transpose", self.transpose_matrix_gui)
        self.create_button(button_row1, "Addition", self.addition_matrix_gui)
        self.create_button(button_row1, "Subtraction", self.subtraction_matrix_gui)
        self.create_button(button_row1, "Multiplication", self.multiply_matrix_gui)

        button_row2 = tk.Frame(self.button_frame, bg=self.button_frame['bg'])
        button_row2.pack(pady=5, fill="x")
        self.create_button(button_row2, "Constant Multiplication", self.cons_mul_gui)
        self.create_button(button_row2, "Exponentiation", self.exp_matrix_gui)
        self.create_button(button_row2, "Diagonal Sum", self.diagonal_sum_gui)
        self.create_button(button_row2, "Symmetric/Skew-Symmetric Check", self.sym_skew_sym_check_gui)

        button_row3 = tk.Frame(self.button_frame, bg=self.button_frame['bg'])
        button_row3.pack(pady=5, fill="x")
        self.create_button(button_row3, "Minors & Cofactors", self.minors_cofactors_gui)
        self.create_button(button_row3, "Determinant (2x2/3x3)", self.determinant_gui)
        self.create_button(button_row3, "Adjoint (2x2/3x3)", self.adjoint_gui)
        self.create_button(button_row3, "Inverse (2x2/3x3)", self.inverse_gui)
        self.create_button(button_row3, "Area of Triangle", self.area_of_triangle_gui)
    
    def create_button(self, parent_frame, text, command):
        btn = tk.Button(parent_frame, text=text, command=command, font=self.button_font, bg=self.button_bg, fg=self.button_fg, 
                        relief="raised", bd=3, padx=10, pady=5)
        btn.pack(side=tk.LEFT, expand=True, fill="x", padx=5, pady=2)
        return btn

    def update_output(self, message):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, message)
        self.output_text.config(state=tk.DISABLED)

    def parse_matrix_input(self, input_string, allow_float=True):
        """Parses a string input from the GUI into a list of lists (matrix)."""
        rows_str = input_string.strip().split('\n')
        matrix = []
        num_cols = -1

        if not rows_str or (len(rows_str) == 1 and not rows_str[0].strip()):
            raise ValueError("Input matrix is empty.")

        for r_idx, row_str in enumerate(rows_str):
            if not row_str.strip(): # Skip empty lines
                continue
            elements_str = row_str.strip().split()
            current_row = []
            
            if not elements_str:
                raise ValueError(f"Row {r_idx + 1} is empty.")

            if num_cols == -1:
                num_cols = len(elements_str)
            elif len(elements_str) != num_cols:
                raise ValueError(f"Inconsistent number of columns. Row {r_idx + 1} has {len(elements_str)} columns, expected {num_cols}.")

            for e_str in elements_str:
                try:
                    if allow_float:
                        val = float(e_str)
                        if val == int(val): # Store as int if it's a whole number
                            current_row.append(int(val))
                        else:
                            current_row.append(val)
                    else:
                        current_row.append(int(e_str))
                except ValueError:
                    raise ValueError(f"Invalid element '{e_str}' in row {r_idx + 1}. Please ensure all elements are valid numbers.")
            matrix.append(current_row)
        
        if not matrix:
            raise ValueError("No valid matrix found in input.")
        return matrix

    def format_matrix_output(self, M):
        """Formats a matrix (list of lists) into a string for display."""
        if not M:
            return "Empty Matrix"
        
        output_str = ""
        # Determine maximum width for each column for aligned printing
        max_widths = [0] * len(M[0])
        for row in M:
            for j, val in enumerate(row):
                max_widths[j] = max(max_widths[j], len(f"{val:.6g}")) # Use .6g for general formatting, limit decimals

        for row in M:
            for j, val in enumerate(row):
                output_str += f"{val:<{max_widths[j] + 2}.6g}" # Add 2 for padding
            output_str += "\n"
        return output_str.strip()

    # --- Matrix Calculation Functions (Adapted from main.py) ---

    def null_matrix(self, r, c):    
        M=[]
        for i in range(r):
            R=[]
            for j in range(c):
                e=0
                R.append(e)
            M.append(R)
        return M

    def transpose_matrix(self, M):     
        r=len(M)
        c=len(M[0])
        A=[]
        for i in range(c):
            B=[]
            for j in range(r):
                B.append(M[j][i])
            A.append(B)
        return A

    def sym_matrix(self, M):           
        r=len(M)
        c=len(M[0])
        A=[]
        for i in range(c):
            B=[]
            for j in range(r):
                B.append(M[j][i])
            A.append(B)
        return A

    def skew_sym_matrix(self, M):      
        r=len(M)
        c=len(M[0])
        A=[]
        for i in range(c):
            B=[]
            for j in range(r):
                B.append((-1)*(M[j][i]))
            A.append(B)
        return A
             
    def addition_matrix(self, M1, M2):      
        r, c = len(M1), len(M1[0])
        A=[]
        for i in range(r):
            B=[]
            for j in range(c):
                C=M1[i][j]+M2[i][j]
                B.append(C)
            A.append(B)
        return A

    def subtraction_matrix(self, M1, M2):   
        r, c = len(M1), len(M1[0])
        A=[]
        for i in range(r):
            B=[]
            for j in range(c):
                C=M1[i][j]-M2[i][j]
                B.append(C)
            A.append(B)
        return A

    def multiply_matrix(self, M1, M2):  
        r1=len(M1)
        c1=len(M1[0])
        r2=len(M2)
        c2=len(M2[0])

        A=[]
        for i in range(r1):
            B=[]
            for j in range(c2): 
                C=0
                for k in range(r2): 
                    C+=M1[i][k]*M2[k][j]
                B.append(C)
            A.append(B)
        return A

    def cons_mul(self, M, n):         
        r=len(M)
        c=len(M[0])
        A=[]
        for i in range(r):
            B=[]
            for j in range(c):
                B.append(n*M[i][j])
            A.append(B)
        return A  

    def exp_matrix(self, M, n):    
        if n < 0:
            return None # Indicate error
        if n == 0:
            size = len(M)
            identity = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
            return identity
        
        result_matrix = [row[:] for row in M] # Deep copy initial matrix
        for _ in range(n - 1):
            size = len(result_matrix)
            result_matrix = self.multiply_matrix(result_matrix, M)
        return result_matrix   

    def diagonal_sum_calc(self, M):     
        trace=0
        r = len(M)
        for i in range(r):
            trace+=M[i][i]
        return trace
        
    def minor_cal(self, M):       
        minors_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        for r_idx in range(3):
            for c_idx in range(3):
                sub_matrix = []
                for i in range(3):
                    if i == r_idx:
                        continue
                    row_elements = []
                    for j in range(3):
                        if j == c_idx:
                            continue
                        row_elements.append(M[i][j])
                    sub_matrix.append(row_elements)
                
                det_sub = (sub_matrix[0][0] * sub_matrix[1][1]) - (sub_matrix[0][1] * sub_matrix[1][0])
                minors_matrix[r_idx][c_idx] = det_sub
                
        return minors_matrix

    def adj_2x2(self, M):          
        adj = [[M[1][1], -M[0][1]], [-M[1][0], M[0][0]]]
        return adj

    def det_2x2(self, M):          
        det=M[0][0]*M[1][1]-M[1][0]*M[0][1]
        return det

    def inverse_2x2(self, M):      
        det = self.det_2x2(M)
        if det == 0:
            return 0 
        
        adj = self.adj_2x2(M)
        
        A=[]
        for i in adj:
            B=[]
            for j in i:
                B.append((1/det)*j)
            A.append(B)
        return A

    def adj_3x3(self, minor_matrix):          
        cofactor_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(3):
            for j in range(3):
                cofactor_matrix[i][j] = ((-1)**(i+j)) * minor_matrix[i][j]
        
        adj = self.transpose_matrix(cofactor_matrix)
        return adj

    def det_3x3(self, M, minor_matrix):     
        cofactor_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(3):
            for j in range(3):
                cofactor_matrix[i][j] = ((-1)**(i+j)) * minor_matrix[i][j]

        det = M[0][0] * cofactor_matrix[0][0] + \
              M[0][1] * cofactor_matrix[0][1] + \
              M[0][2] * cofactor_matrix[0][2]
        return det

    def inverse_3x3(self, M):      
        minor_mat = self.minor_cal(M)
        det = self.det_3x3(M, minor_mat)
        
        if det == 0:
            return 0 
        
        adj_mat = self.adj_3x3(minor_mat)
        
        A=[]        
        for i in adj_mat:
            B=[]
            for j in i:
                B.append((1/det)*j)
            A.append(B)
        return A
            
    def minors_calc(self, minor_matrix):
        Mi1=[]
        for i in range(3):
            Mi2=[]
            for j in range(3):
                Mi2.append(minor_matrix[i][j])
            Mi1.append(Mi2)
        return Mi1 

    def cofactors_calc(self, minor_matrix):
        Co1=[]
        for i in range(3):
            Co2=[]
            for j in range(3):
                x=((-1)**(i+j))*minor_matrix[i][j]
                Co2.append(x)
            Co1.append(Co2)
        return Co1 
        
    def input_for_tri_parse(self, input_string):
        """Parses coordinate input for triangle area. Expects 'x1 y1\\nx2 y2\\nx3 y3'"""
        rows_str = input_string.strip().split('\n')
        if len(rows_str) != 3:
            raise ValueError("Please enter exactly 3 pairs of coordinates (x y) for the triangle vertices, each on a new line.")
        
        A = []
        for i, row_str in enumerate(rows_str):
            elements_str = row_str.strip().split()
            if len(elements_str) != 2:
                raise ValueError(f"Invalid coordinate input in line {i+1}. Expected two numbers (x y).")
            try:
                e1 = float(elements_str[0])
                e2 = float(elements_str[1])
                A.append([e1, e2, 1]) # Add 1 for the 3x3 determinant calculation
            except ValueError:
                raise ValueError(f"Invalid number in line {i+1}. Please ensure coordinates are valid numbers.")
        return A

    # --- GUI Button Handlers ---

    def display_input_matrix(self):
        try:
            input_text = self.input_text.get("1.0", tk.END)
            matrices = input_text.split("MATRIX2")
            
            m1 = self.parse_matrix_input(matrices[0])
            output_message = "Entered Matrix 1:\n" + self.format_matrix_output(m1)
            
            if len(matrices) > 1 and matrices[1].strip():
                m2 = self.parse_matrix_input(matrices[1])
                output_message += "\n\nEntered Matrix 2:\n" + self.format_matrix_output(m2)

            self.update_output(output_message)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def transpose_matrix_gui(self):
        try:
            input_text = self.input_text.get("1.0", tk.END)
            m1 = self.parse_matrix_input(input_text)
            
            transposed_m = self.transpose_matrix(m1)
            output_message = "Original Matrix:\n" + self.format_matrix_output(m1) + \
                             "\n\nTransposed Matrix:\n" + self.format_matrix_output(transposed_m)
            self.update_output(output_message)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def addition_matrix_gui(self):
        try:
            input_text = self.input_text.get("1.0", tk.END)
            matrices = input_text.split("MATRIX2")
            if len(matrices) < 2 or not matrices[1].strip():
                raise ValueError("Please enter two matrices separated by 'MATRIX2' for addition.")
            
            m1 = self.parse_matrix_input(matrices[0])
            m2 = self.parse_matrix_input(matrices[1])

            if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
                raise ValueError("Addition not possible: Matrices must have the same dimensions.")
            
            result_m = self.addition_matrix(m1, m2)
            output_message = "Matrix 1:\n" + self.format_matrix_output(m1) + \
                             "\n\nMatrix 2:\n" + self.format_matrix_output(m2) + \
                             "\n\nResult of Addition:\n" + self.format_matrix_output(result_m)
            self.update_output(output_message)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def subtraction_matrix_gui(self):
        try:
            input_text = self.input_text.get("1.0", tk.END)
            matrices = input_text.split("MATRIX2")
            if len(matrices) < 2 or not matrices[1].strip():
                raise ValueError("Please enter two matrices separated by 'MATRIX2' for subtraction.")
            
            m1 = self.parse_matrix_input(matrices[0])
            m2 = self.parse_matrix_input(matrices[1])

            if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
                raise ValueError("Subtraction not possible: Matrices must have the same dimensions.")
            
            result_m = self.subtraction_matrix(m1, m2)
            output_message = "Matrix 1:\n" + self.format_matrix_output(m1) + \
                             "\n\nMatrix 2:\n" + self.format_matrix_output(m2) + \
                             "\n\nResult of Subtraction:\n" + self.format_matrix_output(result_m)
            self.update_output(output_message)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def multiply_matrix_gui(self):
        try:
            input_text = self.input_text.get("1.0", tk.END)
            matrices = input_text.split("MATRIX2")
            if len(matrices) < 2 or not matrices[1].strip():
                raise ValueError("Please enter two matrices separated by 'MATRIX2' for multiplication.")
            
            m1 = self.parse_matrix_input(matrices[0])
            m2 = self.parse_matrix_input(matrices[1])

            if len(m1[0]) != len(m2):
                raise ValueError("Multiplication not possible: Number of columns of Matrix 1 must equal number of rows of Matrix 2.")
            
            result_m = self.multiply_matrix(m1, m2)
            output_message = "Matrix 1:\n" + self.format_matrix_output(m1) + \
                             "\n\nMatrix 2:\n" + self.format_matrix_output(m2) + \
                             "\n\nResult of Multiplication:\n" + self.format_matrix_output(result_m)
            self.update_output(output_message)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def cons_mul_gui(self):
        try:
            input_text = self.input_text.get("1.0", tk.END)
            m1 = self.parse_matrix_input(input_text)

            scalar_str = simpledialog.askstring("Constant Input", "Enter the constant to multiply by:")
            if scalar_str is None: # User cancelled
                return
            
            try:
                scalar = float(scalar_str)
            except ValueError:
                raise ValueError("Invalid constant. Please enter a number.")

            result_m = self.cons_mul(m1, scalar)
            output_message = "Original Matrix:\n" + self.format_matrix_output(m1) + \
                             f"\n\nMultiplied by Constant: {scalar}\n\nResult:\n" + self.format_matrix_output(result_m)
            self.update_output(output_message)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def exp_matrix_gui(self):
        try:
            input_text = self.input_text.get("1.0", tk.END)
            m1 = self.parse_matrix_input(input_text)

            r, c = len(m1), len(m1[0])
            if r != c:
                raise ValueError("Exponentiation is only possible for square matrices.")

            exponent_str = simpledialog.askinteger("Exponent Input", "Enter the integer exponent (non-negative):", minvalue=0)
            if exponent_str is None: # User cancelled
                return
            
            exponent = int(exponent_str) # simpledialog.askinteger returns int or None

            result_m = self.exp_matrix(m1, exponent)
            if result_m is None: # Error from exp_matrix (e.g., negative exponent)
                raise ValueError("Exponentiation with negative power is not supported.")

            output_message = "Original Matrix:\n" + self.format_matrix_output(m1) + \
                             f"\n\nRaised to the power of: {exponent}\n\nResult:\n" + self.format_matrix_output(result_m)
            self.update_output(output_message)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def diagonal_sum_gui(self):
        try:
            input_text = self.input_text.get("1.0", tk.END)
            m1 = self.parse_matrix_input(input_text)
            
            r, c = len(m1), len(m1[0])
            if r != c:
                raise ValueError("Diagonal sum is only applicable for square matrices.")
            
            sum_diag = self.diagonal_sum_calc(m1)
            output_message = "Original Matrix:\n" + self.format_matrix_output(m1) + \
                             f"\n\nDiagonal Sum: {sum_diag}"
            self.update_output(output_message)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def sym_skew_sym_check_gui(self):
        try:
            input_text = self.input_text.get("1.0", tk.END)
            m1 = self.parse_matrix_input(input_text)

            r, c = len(m1), len(m1[0])
            if r != c:
                raise ValueError("Symmetry checking is only applicable for square matrices.")
            
            m_transpose = self.transpose_matrix(m1)
            m_neg_transpose = self.skew_sym_matrix(m1) # This actually returns -M^T
            m_null = self.null_matrix(r, c)

            result_message = "Original Matrix:\n" + self.format_matrix_output(m1) + "\n\n"

            if m1 == m_null:
                result_message += "Entered Matrix is a **SYMMETRIC** and a **SKEW-SYMMETRIC** Matrix (as it is a NULL Matrix)"
            elif m1 == m_transpose:
                result_message += "Entered Matrix is a **SYMMETRIC** Matrix"
            elif m1 == m_neg_transpose:
                result_message += "Entered Matrix is a **SKEW-SYMMETRIC** Matrix"
            else:
                result_message += "Entered Matrix is neither a **SYMMETRIC** nor a **SKEW-SYMMETRIC** Matrix"
            
            self.update_output(result_message)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def minors_cofactors_gui(self):
        try:
            input_text = self.input_text.get("1.0", tk.END)
            m1 = self.parse_matrix_input(input_text)

            r, c = len(m1), len(m1[0])
            if r != 3 or c != 3:
                raise ValueError("Minors and Cofactors calculation is only supported for 3x3 matrices.")
            
            minor_mat = self.minor_cal(m1)
            cofactor_mat = self.cofactors_calc(minor_mat) # This function actually calculates cofactors from minors
            
            output_message = "Original Matrix:\n" + self.format_matrix_output(m1) + \
                             "\n\nMinors Matrix:\n" + self.format_matrix_output(minor_mat) + \
                             "\n\nCofactors Matrix:\n" + self.format_matrix_output(cofactor_mat)
            self.update_output(output_message)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def determinant_gui(self):
        try:
            input_text = self.input_text.get("1.0", tk.END)
            m1 = self.parse_matrix_input(input_text)

            r, c = len(m1), len(m1[0])
            
            det = None
            if r == 2 and c == 2:
                det = self.det_2x2(m1)
            elif r == 3 and c == 3:
                minor_mat = self.minor_cal(m1)
                det = self.det_3x3(m1, minor_mat)
            else:
                raise ValueError("Determinant calculation is only supported for 2x2 or 3x3 matrices.")
            
            output_message = "Original Matrix:\n" + self.format_matrix_output(m1) + \
                             f"\n\nDeterminant: {det}"
            self.update_output(output_message)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def adjoint_gui(self):
        try:
            input_text = self.input_text.get("1.0", tk.END)
            m1 = self.parse_matrix_input(input_text)

            r, c = len(m1), len(m1[0])
            
            adj_m = None
            if r == 2 and c == 2:
                adj_m = self.adj_2x2(m1)
            elif r == 3 and c == 3:
                minor_mat = self.minor_cal(m1)
                adj_m = self.adj_3x3(minor_mat)
            else:
                raise ValueError("Adjoint calculation is only supported for 2x2 or 3x3 matrices.")
            
            output_message = "Original Matrix:\n" + self.format_matrix_output(m1) + \
                             "\n\nAdjoint Matrix:\n" + self.format_matrix_output(adj_m)
            self.update_output(output_message)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def inverse_gui(self):
        try:
            input_text = self.input_text.get("1.0", tk.END)
            m1 = self.parse_matrix_input(input_text)

            r, c = len(m1), len(m1[0])
            
            inverse_m = None
            if r == 2 and c == 2:
                inverse_m = self.inverse_2x2(m1)
            elif r == 3 and c == 3:
                inverse_m = self.inverse_3x3(m1)
            else:
                raise ValueError("Inverse calculation is only supported for 2x2 or 3x3 matrices.")
            
            if inverse_m == 0:
                output_message = "Original Matrix:\n" + self.format_matrix_output(m1) + \
                                 "\n\nInverse Matrix does not exist as Determinant is 0."
            else:
                output_message = "Original Matrix:\n" + self.format_matrix_output(m1) + \
                                 "\n\nInverse Matrix:\n" + self.format_matrix_output(inverse_m)
            self.update_output(output_message)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def area_of_triangle_gui(self):
        try:
            input_text = self.input_text.get("1.0", tk.END)
            # This function expects 'x1 y1\nx2 y2\nx3 y3'
            m1 = self.input_for_tri_parse(input_text) 
            
            minor_mat = self.minor_cal(m1)
            area = (1/2) * (self.det_3x3(m1, minor_mat))
            final_area = abs(area)

            output_message = "Entered Vertices Matrix:\n" + self.format_matrix_output(m1) + \
                             f"\n\nAREA of the TRIANGLE is : {final_area:.4f} sq.units"
            self.update_output(output_message)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MatrixCalculatorGUI(root)
    root.mainloop()

