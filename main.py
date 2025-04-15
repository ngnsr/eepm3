import numpy as np
import decimal
from decimal import Decimal
import sys
import os
from scipy import linalg
# import asyncio

class Results:
    def __init__(self, eigenvalues, b, c, d, frobenius, right_frobenius_vec, 
                 left_frobenius_vec, is_productive, matrix_b=None, 
                 convergence_n=None, price_vector=None):
        self.eigenvalues = eigenvalues
        self.b = b
        self.c = c
        self.d = d
        self.frobenius = frobenius
        self.right_frobenius_vec = right_frobenius_vec
        self.left_frobenius_vec = left_frobenius_vec
        self.is_productive = is_productive
        self.matrix_b = matrix_b
        self.convergence_n = convergence_n
        self.price_vector = price_vector

def read_matrix_from_file(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Файл "{file_path}" не знайдено')
        
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        
        matrix_data = []
        for line in file_content.strip().split('\n'):
            values = [float(val) for val in line.strip().split()]
            if not all(isinstance(x, (int, float)) for x in values):
                raise ValueError(f'Некоректні дані у файлі "{file_path}": рядок містить нечислові значення')
            matrix_data.append(values)
        
        row_count = len(matrix_data)
        if row_count == 0:
            raise ValueError(f'Файл "{file_path}" порожній')
        
        col_count = len(matrix_data[0])
        if any(len(row) != col_count for row in matrix_data):
            raise ValueError(f'Некоректні дані у файлі "{file_path}": матриця повинна бути квадратною')
        
        if row_count != col_count:
            raise ValueError(f'Некоректні дані у файлі "{file_path}": матриця повинна бути квадратною')
        
        return matrix_data
    
    except Exception as error:
        print(f'Помилка при читанні матриці з файлу: {str(error)}')
        sys.exit(1)

def read_vector_from_file(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Файл "{file_path}" не знайдено')
        
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        
        vector_data = []
        lines = file_content.strip().split('\n')
        if lines:
            vector_data = [float(val) for val in lines[0].strip().split()]
            if not all(isinstance(x, (int, float)) for x in vector_data):
                raise ValueError(f'Некоректні дані у файлі "{file_path}": рядок містить нечислові значення')
        
        return vector_data
    
    except Exception as error:
        print(f'Помилка при читанні вектору з файлу: {str(error)}')
        sys.exit(1)

# Функції для аналізу матриці
def analyze_matrix(matrix_data, vector_data, accuracy):
    try:
        A = np.array(matrix_data, dtype=float)
        eigenvalues, right_eigenvectors = linalg.eig(A)
        
        # Обчислюємо ліві власні вектори через транспоновану матрицю
        _, left_eigenvectors = linalg.eig(A.T)
        
        n = len(eigenvalues)
        lambdas = [Decimal(str(val.real)) for val in eigenvalues]
        
        # Знаходимо коефіцієнти характеристичного полінома
        b = -sum(lambdas)
        
        if n == 3:
            lambda1, lambda2, lambda3 = lambdas
            c = lambda1 * lambda2 + lambda1 * lambda3 + lambda2 * lambda3
            d = -(lambda1 * lambda2 * lambda3)
        else:
            c = Decimal('0')
            d = Decimal('0')
        
        # Знаходимо число Фробеніуса
        frobenius = max(abs(val) for val in eigenvalues)
        fro_index = list(abs(val) for val in eigenvalues).index(frobenius)
        
        right_frobenius_vec = right_eigenvectors[:, fro_index].real
        left_frobenius_vec = left_eigenvectors[:, fro_index].real
        
        # Нормалізація власних векторів
        right_frobenius_vec = right_frobenius_vec / np.sum(right_frobenius_vec)
        left_frobenius_vec = left_frobenius_vec / np.sum(left_frobenius_vec)
        
        is_productive = all(abs(val) < 1 for val in eigenvalues)
        
        # Обчислюємо матрицю повних витрат і збіжність ряду
        B, N = build_full_cost_matrix(A, accuracy)
        
        # Обчислюємо вектор цін
        price_vector = calculate_price_vector(B, vector_data)
        
        return Results(
            eigenvalues=[val.real for val in eigenvalues],
            b=b,
            c=c,
            d=d,
            frobenius=frobenius,
            right_frobenius_vec=right_frobenius_vec,
            left_frobenius_vec=left_frobenius_vec,
            is_productive=is_productive,
            matrix_b=B,
            convergence_n=N,
            price_vector=price_vector
        )
    
    except Exception as error:
        print(f'Помилка при аналізі матриці: {str(error)}')
        sys.exit(1)

def build_full_cost_matrix(A, epsilon=0.01):
    n = A.shape[0]
    E = np.eye(n)
    E_minus_A = E - A
    B = np.linalg.inv(E_minus_A)
    
    current_sum = E.copy()
    N = 0
    max_diff = float('inf')
    
    while max_diff >= epsilon:
        N += 1
        A_power_N = np.linalg.matrix_power(A, N)
        prev_sum = current_sum.copy()
        current_sum = current_sum + A_power_N
        max_diff = np.max(np.abs(current_sum - prev_sum))
    
    return B, N

def calculate_price_vector(B, s):
    s_vector = np.array(s).reshape(-1, 1)
    BT = B.T
    p_vector = BT.dot(s_vector)
    return p_vector.flatten()

# Функція для відображення результатів
def display_results(results):
    print("\n--- Характеристичний поліном ---")
    print(f"λ^3 + ({results.b:.6f})λ^2 + ({results.c:.6f})λ + ({results.d:.6f}) = 0")
    
    print("\n--- Власні значення ---")
    for i, λ in enumerate(results.eigenvalues):
        print(f"λ{i+1} = {λ:.6f}")
    
    print("\n--- Число Фробеніуса ---")
    print(f"{results.frobenius:.6f}")
    
    print("\n--- Правий вектор Фробеніуса ---")
    print(f"({', '.join([f'{x:.6f}' for x in results.right_frobenius_vec])})")
    
    print("\n--- Лівий вектор Фробеніуса ---")
    print(f"({', '.join([f'{x:.6f}' for x in results.left_frobenius_vec])})")
    
    print("\n--- Висновок ---")
    print("Матриця є продуктивною (усі |λ| < 1)" if results.is_productive else "Матриця НЕ є продуктивною")
    
    print("\n--- Матриця повних витрат і збіжність ряду ---")
    if results.matrix_b is not None:
        print("\nМатриця повних витрат B = (E - A)^(-1):")
        for row in results.matrix_b:
            print(f"[{'\t'.join([f'{x:.6f}' for x in row])}]")
        
        print(f"\nДля збіжності ряду E + A + A^2 + ... + A^N з точністю 0.01 необхідно N = {results.convergence_n}")
    
    print("\n--- Вектор цін ---")
    if results.price_vector is not None:
        print(f"({', '.join([f'{x:.6f}' for x in results.price_vector])})")

def main():
    matrix_file_path = sys.argv[1] if len(sys.argv) > 1 else "matrix.txt"
    vector_file_path = sys.argv[2] if len(sys.argv) > 2 else "vector.txt"
    
    try:
        matrix_data = read_matrix_from_file(matrix_file_path)
        vector_data = read_vector_from_file(vector_file_path)
        
        print("\n--- Вхідна матриця ---")
        for row in matrix_data:
            print("\t".join(str(val) for val in row))
        
        print("\n--- Вектор ---")
        print(f"({', '.join(str(val) for val in vector_data)})")
        
        # Використовуємо asyncio для отримання введення користувача
        accuracy_input = input("\nВведіть бажану точність для аналізу збіжності ряду: ")
        accuracy = float(accuracy_input)
        
        if np.isnan(accuracy):
            raise ValueError("Некоректно введена точність")
        
        results = analyze_matrix(matrix_data, vector_data, accuracy)
        display_results(results)
    
    except Exception as error:
        print(f"Помилка при обробці матриці: {str(error)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
