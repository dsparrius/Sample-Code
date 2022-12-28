import numpy as np
import itertools

def main():
    a = get_matrix()
    dim = len(a)
    ones = np.ones(dim)
    iden = np.diag(ones)

    eigen = np.linalg.eigvals(a)

    #creates dictionary pairing eigen values to thier multiplicities
    eigen_list =[x for x in eigen]
    roots = []
    [roots.append(x) for x in eigen if x not in roots]
    multiplicities = {x:eigen_list.count(x) for x in roots}

    #creates a tuple of multiplicites whose sum adds up to the degree of the characteristic polynomial
    char_degree = tuple(multiplicities[x] for x in roots)

    #Finds the multiplicities of the eigen values in the minimal polynomial
    factors = []
    for _ in roots:
        factors.append(a-_*iden)

    #gets possible multiplicies for the invariant factors based on multiplicities in characteristic polynomial
    degrees = []
    for _ in roots:
        degrees.append([i for i in range(1,multiplicities[_]+1)])

    #creates all possible combinations of multiplicities
    comb = itertools.product(*degrees)
    
    min_degree = char_degree
    for _ in comb:
        m = iden
        for i in range(len(factors)):
            m = m@np.linalg.matrix_power(factors[i],_[i])

        #checks to see if the combination of of degrees as powers yield zero matrix
        #then assigns degree of minimal polynomial if zero matrix is confirmed
        if np.linalg.norm(m,ord = np.inf) == 0:
            min_degree = _
            break
        else:
            pass

    elementary_factors = []

    #handles case where minimal polynomial is identical to characteristic polynomail
    #gives roots and degrees in elementary factors
    if min_degree == char_degree:
        for i in range(len(roots)):
            elementary_factors.append([roots[i],min_degree[i]])

    #handles more common case of degree of minimal polynomial < degree characteristic polynomial
    else:
        invarfactors = get_invariant_factors(char_degree,min_degree)

        #Checks to see if there is ambiguity in possible invariant factors
        #i.e. same minimal and characteristic polynomials but different invariant factors
        if len(invarfactors) != 1:
            invarfactors = get_unique_invariant(a,invarfactors)
        factors = [min_degree]
        for _ in invarfactors:
            factors.append(_[0])

        for i in range(len(roots)):
            for y in factors:
                if y[i] != 0:
                    elementary_factors.append([roots[i],y[i]])
                else:
                    pass

    #creates collection of Jordan Blocks
    jordan_blocks  = []
    for _ in elementary_factors:
        jordan = get_jordan_block(_)
        jordan_blocks.append(jordan)


    jordan_form = get_jordan_form(char_degree,jordan_blocks)
    print(jordan_form)


#Gets matrix from user. Starts with dimension of the square matrix then asks for each row.
def get_matrix():
    rows = []
    while True:
        try:
            size = int(input('How many rows are in the square matrix? '))
            break
        except ValueError:
            print('Please enter an integer.')

    n = 0
    print('Enter each row seperating each entry by a comma:')
    while n < size:
        row = input().split(',')
        try:
            for i in range(size):
                row[i] = complex(row[i])
            rows.append(row)
            n+=1
        except ValueError:
            print('Enter the row again in the correct format:')
        except IndexError:
            print(f'Enter the row again with {size} enteries:')
    return np.array(rows)

#Determind the invariant factors based on the degrees of the characteristic polynomial and minimal polynomial
def get_invariant_factors(char_degree, min_degree):
    degrees = []
    for _ in min_degree:
        degrees.append([i for i in range(0,_+1)])

    comb= []
    for _ in itertools.product(*degrees):
        comb.append(_)

    trivial_degree = comb[0]


    i = char_degree[0] - min_degree[0]
    potential_polys = []

    for _ in range(1,i+1):
        for x in itertools.product(comb,repeat = _):
            potential_polys.append(x)


    a = 0
    while a < len(min_degree):
        i = char_degree[a] - min_degree[a]
        candidate = []
        for _ in potential_polys:
            if _ == trivial_degree or trivial_degree in _:
                pass
            else:
                l = len(_)
                j = 0
                degree = 0
                while j < l:
                    d = _[j]
                    j+=1
                    degree += d[a]

                if degree == i:
                    candidate.append(_)
                else:
                    pass
        potential_polys = candidate
        a+=1

    potential_invariants = []
    for _ in potential_polys:
        temp_list = []
        for x in _:
            temp_list.append(x)
        potential_invariants.append(temp_list)
    return potential_invariants

#Handles cases where there are same characteristic and minimal polynomials but multiple possibilities for invariant factors
#ex: invariant factors x,x,x^2 so that x*x*x^2 = x^4 while invariant factors x^2,x^2 so that x^2*x^2 = x^4
def get_unique_invariant(a,potential_invariant):
    eigval,eigvec = np.linalg.eig(a)
    eigdim = 0
    for _ in eigvec:
        if np.linalg.norm(_) != 0:
            eigdim+=1
        else:
            pass
    for _ in potential_invariant:
        if len(_) != eigdim -1:
            pass
        else:
            return [_]

#Determines the Jordan Block based on the given elementary factor
def get_jordan_block(elementary_factor):
    dim = elementary_factor[1]
    if dim == 1:
        emptyjordan = np.array([elementary_factor[0]])
    else:
        emptyjordan = np.zeros(dim*dim).reshape(dim,dim)
        position = 0
        while position < dim:
            if position == 0:
                emptyjordan[position,position] = elementary_factor[0]
            else:
                emptyjordan[position,position] = elementary_factor[0]
                emptyjordan[position-1,position] = 1

            position +=1
    return emptyjordan

#Returns the Jordan Canonical Form from the Jordan Blocks
def get_jordan_form(char_degrees,jordan_blocks):
    dim = sum(list(char_degrees))
    jordanform = np.zeros(dim*dim, dtype = complex).reshape(dim,dim)

    #Keeps track of position after each block has been copied into the matrix
    position = 0

    #copies the jordan blocks into the matrix
    for _ in jordan_blocks:
        size = _.ndim
        row,column = 0,0
        while row < size:
            while column < size:
                if size == 1:
                    jordanform[column+position,row + position] = _[column]
                else:
                    jordanform[column+position,row + position] = _[column,row]
                column+=1

            row+=1
            column = 0

        #adjusts position to be in the correct spot for the next Jordan Block
        position+=row

    return jordanform


if __name__ == '__main__':
    main()