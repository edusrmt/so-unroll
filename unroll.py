import os
import threading
import random

"""
========================================================================================

 ######  ########  #######  ##     ## ######## ##    ## ######## ####    ###    ##       
##    ## ##       ##     ## ##     ## ##       ###   ##    ##     ##    ## ##   ##       
##       ##       ##     ## ##     ## ##       ####  ##    ##     ##   ##   ##  ##       
 ######  ######   ##     ## ##     ## ######   ## ## ##    ##     ##  ##     ## ##       
      ## ##       ##  ## ## ##     ## ##       ##  ####    ##     ##  ######### ##       
##    ## ##       ##    ##  ##     ## ##       ##   ###    ##     ##  ##     ## ##       
 ######  ########  ##### ##  #######  ######## ##    ##    ##    #### ##     ## ########
 
========================================================================================
"""

"""
.--------.-------------------------.
| mtxA   | First matrix            |
:--------+-------------------------:
| mtxB   | Second matrix           |
:--------+-------------------------:
| RETURN | Matrix of the summation |
'--------'-------------------------'
"""

def sum_matrix (mtxA, mtxB):
    mtxC = [[0 for i in range(MATRIX_SIZE)] for i in range(MATRIX_SIZE)]

    for i in range(n):
        for j in range(n):
            mtxC[i][j] = mtxA[i][j] + mtxB[i][j]

    return mtxC

"""
.--------.-----------------------.
| mtxA   | First matrix          |
:--------+-----------------------:
| mtxB   | Second matrix         |
:--------+-----------------------:
| RETURN | Matrix of the product |
'--------'-----------------------'
"""

def prod_matrix (mtxA, mtxB):
    mtxC = [[0 for i in range(MATRIX_SIZE)] for i in range(MATRIX_SIZE)]

    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            for k in range(MATRIX_SIZE):
                mtxC[i][j] += mtxA[i][k] * mtxB[k][j]

    return mtxC

"""
===========================================================================

########     ###    ########     ###    ##       ##       ######## ##       
##     ##   ## ##   ##     ##   ## ##   ##       ##       ##       ##       
##     ##  ##   ##  ##     ##  ##   ##  ##       ##       ##       ##       
########  ##     ## ########  ##     ## ##       ##       ######   ##       
##        ######### ##   ##   ######### ##       ##       ##       ##       
##        ##     ## ##    ##  ##     ## ##       ##       ##       ##       
##        ##     ## ##     ## ##     ## ######## ######## ######## ######## 

===========================================================================
"""

"""
.--------.------------------------------------------------.
| val1   | Value of first matrix                          |
:--------+------------------------------------------------:
| val2   | Value of second matrix                         |
:--------+------------------------------------------------:
| i      | Row index                                      |
:--------+------------------------------------------------:
| j      | Column index                                   |
:--------+------------------------------------------------:
| res    | Reference to store the result of the operation |
'--------'------------------------------------------------'
"""

def add (val1, val2, i, j, res):
    res[i][j] = val1 + val2

"""
.-----.------------------------------------------------.
| row | Row of the first matrix                        |
:-----+------------------------------------------------:
| col | Column of the second matrix                    |
:-----+------------------------------------------------:
| i   | Row index                                      |
:-----+------------------------------------------------:
| j   | Column index                                   |
:-----+------------------------------------------------:
| res | Reference to store the result of the operation |
'-----'------------------------------------------------'
"""

def prod (row, col, i, j, res):
    for k in range(len(row)):
        res[i][j] += row[k] * col[k]

"""
========================================================

##     ## ##    ## ########   #######  ##       ##       
##     ## ###   ## ##     ## ##     ## ##       ##       
##     ## ####  ## ##     ## ##     ## ##       ##       
##     ## ## ## ## ########  ##     ## ##       ##       
##     ## ##  #### ##   ##   ##     ## ##       ##       
##     ## ##   ### ##    ##  ##     ## ##       ##       
 #######  ##    ## ##     ##  #######  ######## ######## 

========================================================
"""

def unroll (args, func, results):
    for i in range(len(args)):
        arg = args[i]
        th = threading.Thread(target=func, args=(*arg, results))
        th.start()

"""
=======================================================

##     ## ######## #### ##       #### ######## ##    ## 
##     ##    ##     ##  ##        ##     ##     ##  ##  
##     ##    ##     ##  ##        ##     ##      ####   
##     ##    ##     ##  ##        ##     ##       ##    
##     ##    ##     ##  ##        ##     ##       ##    
##     ##    ##     ##  ##        ##     ##       ##    
 #######     ##    #### ######## ####    ##       ##    

=======================================================
"""

"""
.--------.--------------------------------------.
| mtx    | Origin matrix                        |
:--------+--------------------------------------:
| j      | Desired column                       |
:--------+--------------------------------------:
| RETURN | List with all values from the column |
'--------'--------------------------------------'
"""

def get_column (mtx, j):
    col = [0 for i in range(MATRIX_SIZE)]

    for i in range(MATRIX_SIZE):
        col[i] = mtx[i][j]

    return col


"""
.--------.--------------------------------.
| n      | Dimension of the square matrix |
:--------+--------------------------------:
| max    | Limit for the values           |
:--------+--------------------------------:
| RETURN | Matrix with random numbers     |
'--------'--------------------------------'
"""

def random_matrix (n, max):
    mtx = [[0 for i in range(n)] for i in range(n)]

    for i in range(n):
        for j in range(n):
            mtx[i][j] = random.randint(0, max)

    return mtx

"""
.--------.-----------------------------------------.
| mtxA   | First matrix                            |
:--------+-----------------------------------------:
| mtxB   | Second matrix                           |
:--------+-----------------------------------------:
| RETURN | Unroll compatible matrix with arguments |
'--------'-----------------------------------------'
"""

def generate_add_args (mtxA, mtxB):
    args = [[0 for i in range(4)] for i in range(MATRIX_SIZE * MATRIX_SIZE)]

    k = 0
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            args[k] = [mtxA[i][j], mtxB[i][j], i, j]
            k += 1

    return args

"""
.--------.-----------------------------------------.
| mtxA   | First matrix                            |
:--------+-----------------------------------------:
| mtxB   | Second matrix                           |
:--------+-----------------------------------------:
| RETURN | Unroll compatible matrix with arguments |
'--------'-----------------------------------------'
"""

def generate_prod_args (mtxA, mtxB):
    args = [[0 for i in range(4)] for i in range(MATRIX_SIZE * MATRIX_SIZE)]

    k = 0
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            args[k] = [mtxA[i], get_column(mtxB, j), i, j]
            k += 1

    return args


"""
.-------.----------------------.
| mtx   | Matrix to be printed |
:-------+----------------------:
| PRINT | Stylized matrix      |
'-------'----------------------'
"""

def print_matrix (mtx):
    print('|\t', end='')
    print('\t|\n|\t'.join(['\t'.join([str(cell) for cell in row]) for row in mtx]), end='')
    print('\t|')

"""
=================================

##     ##    ###    #### ##    ## 
###   ###   ## ##    ##  ###   ## 
#### ####  ##   ##   ##  ####  ## 
## ### ## ##     ##  ##  ## ## ## 
##     ## #########  ##  ##  #### 
##     ## ##     ##  ##  ##   ### 
##     ## ##     ## #### ##    ## 

=================================                                          
"""

# GERAR DUAS MATRIZES ALEATÓRIAS
MATRIX_SIZE = 3
MAX_VALUE = 10

matA = random_matrix(MATRIX_SIZE, MAX_VALUE)
matB = random_matrix(MATRIX_SIZE, MAX_VALUE)

# IMPRIMIR AS MATRIZES INICIAIS
print("""
 __    __     ______     ______   ______     __     __  __        ______    
/\ "-./  \   /\  __ \   /\__  _\ /\  == \   /\ \   /\_\_\_\      /\  __ \   
\ \ \-./\ \  \ \  __ \  \/_/\ \/ \ \  __<   \ \ \  \/_/\_\/_     \ \  __ \  
 \ \_\ \ \_\  \ \_\ \_\    \ \_\  \ \_\ \_\  \ \_\   /\_\/\_\     \ \_\ \_\ 
  \/_/  \/_/   \/_/\/_/     \/_/   \/_/ /_/   \/_/   \/_/\/_/      \/_/\/_/                                                                             
""")
print_matrix(matA)

print("""
 __    __     ______     ______   ______     __     __  __        ______    
/\ "-./  \   /\  __ \   /\__  _\ /\  == \   /\ \   /\_\_\_\      /\  == \   
\ \ \-./\ \  \ \  __ \  \/_/\ \/ \ \  __<   \ \ \  \/_/\_\/_     \ \  __<   
 \ \_\ \ \_\  \ \_\ \_\    \ \_\  \ \_\ \_\  \ \_\   /\_\/\_\     \ \_____\ 
  \/_/  \/_/   \/_/\/_/     \/_/   \/_/ /_/   \/_/   \/_/\/_/      \/_____/                                                                             
""")
print_matrix(matB)

# IMPRIMIR AS MATRIZES DAS OPERAÇÕES

print("""
 ______     _____     _____     __     ______   __     ______     __   __    
/\  __ \   /\  __-.  /\  __-.  /\ \   /\__  _\ /\ \   /\  __ \   /\ "-.\ \   
\ \  __ \  \ \ \/\ \ \ \ \/\ \ \ \ \  \/_/\ \/ \ \ \  \ \ \/\ \  \ \ \-.  \  
 \ \_\ \_\  \ \____-  \ \____-  \ \_\    \ \_\  \ \_\  \ \_____\  \ \_\ \ _\ 
  \/_/\/_/   \/____/   \/____/   \/_/     \/_/   \/_/   \/_____/   \/_/ \/_/ 
""")

matC = [[0 for i in range(MATRIX_SIZE)] for i in range(MATRIX_SIZE)]
args = generate_add_args(matA, matB)
unroll(args, add, matC)
print_matrix(matC)

print("""
 ______   ______     ______     _____     __  __     ______     ______  
/\  == \ /\  == \   /\  __ \   /\  __-.  /\ \/\ \   /\  ___\   /\__  _\ 
\ \  _-/ \ \  __<   \ \ \/\ \  \ \ \/\ \ \ \ \_\ \  \ \ \____  \/_/\ \/ 
 \ \_\    \ \_\ \_\  \ \_____\  \ \____-  \ \_____\  \ \_____\    \ \_\ 
  \/_/     \/_/ /_/   \/_____/   \/____/   \/_____/   \/_____/     \/_/ 
""")

matC = [[0 for i in range(MATRIX_SIZE)] for i in range(MATRIX_SIZE)]
args = generate_prod_args(matA, matB)
unroll(args, prod, matC)
print_matrix(matC)