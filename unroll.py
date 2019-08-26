"""
|
|
| func(arg1, arg2, ..., argX)
|
|
V

-------------------> 
arg1 arg2 ... argX

func(0,1)
func(1,2)
func(2,3)
func(3,4)
func(4,5)

0 1
1 2
2 3
3 4
4 5

unroll
	[[0,1], [1,2], [2,3], [3,4], [4,5]]
	func
	results -----> [1, 3, 5, 7, 9]
	method  -----> 'proc' ou 'thre'
"""

# Soma de duas matrizes
def sum (matrixA, matrixB):
    return 0

# Multiplicação de duas matrizes
def multiplication (matrixA, matrixB):
    return 0

"""
args    --> Matriz com os parâmentros da função alvo
func    --> Função a ser executada em paralelo
results --> Vetor com os retornos da função, caso haja
method  --> Tipo de implementação paralela a ser utiizada
"""
def unroll (args, func, results, method):
    return 0

# PROGRAMA PRINCIPAL
matrixA = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
matrixB = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]

print("MATRIX A:")
for a in matrixA:
    print(a)

print(" ")

print("MATRIX B:")
for b in matrixB:
    print(b)