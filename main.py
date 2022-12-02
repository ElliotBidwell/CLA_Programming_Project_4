# Elliot Bidwell
# CS 2300 - 001
# Programming Project 4

# Sources:
# https://www.geeksforgeeks.org/python-program-multiply-two-matrices/ for an example algorithm for multiplying
# matrices. I generalized it to work with N x N matrices.

from math import *


class Planes_Points_Lines:

    def __init__(self):
        self.plane = []
        self.direction = []

        self.points = []
        self.planes_and_points = []

        self.line = []
        self.triangles = []

        self.stochastic = []

    # These function collect values from input files to be used in other functions
    def get_input_projection(self, filename):
        # Reset points list
        self.points = []

        input_file = open(filename, 'r', encoding="utf-8")
        line_1 = input_file.readline().split()
        self.plane = [[float(line_1[i]) for i in range(3)], [float(line_1[i]) for i in range(3, 6)]]
        self.direction = [float(line_1[i]) for i in range(6, 9)]

        for line in input_file:
            self.points.extend([[float(line.split()[i]) for i in range( j * 3,(j + 1) * 3)] for j in range(3)])

        input_file.close()

    def get_input_distance(self, filename):
        # Reset planes_and_points list
        self.planes_and_points = []

        input_file = open(filename, 'r', encoding="utf-8")

        for line in input_file:
            self.planes_and_points.extend([[[float(line.split()[i]) for i in range(3)], [float(line.split()[i]) for i in range(3, 6)], [float(line.split()[i]) for i in range(6, 9)]]])

        input_file.close()

    def get_input_intersection(self, filename):
        # Reset triangles list
        self.triangles = []
        input_file = open(filename, 'r', encoding="utf-8")

        line_1 = input_file.readline().split()
        self.line = [[float(line_1[i]) for i in range(3)], [float(line_1[i]) for i in range(3, 6)]]

        for line in input_file:
            self.triangles.extend([[[float(line.split()[i]) for i in range(3)], [float(line.split()[i]) for i in range(3, 6)], [float(line.split()[i]) for i in range(6, 9)]]])

        input_file.close()

    def get_input_stochastic(self, filename):
        # Reset stochastic list
        self.stochastic = []

        input_file = open(filename, 'r', encoding="utf-8")

        for line in input_file:
            self.stochastic.append([float(x) for x in line.split()])

        input_file.close()

    # This function returns the normalized form of a vector
    def do_normalize(self, vector):
        normalized_vector = []

        sum_of_squares = 0
        for x in vector:
            sum_of_squares += x ** 2
        magnitude = sqrt(sum_of_squares)

        x: object
        for x in vector:
            normalized_vector.append(x/magnitude)

        return normalized_vector

    # This function uses equation (10.8) from the textbook to perform a parallel projection
    def do_parallel_projection(self, filename):
        plane_point = self.plane[0]
        plane_normal = self.do_normalize(self.plane[1])
        proj_direction = self.direction
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        vn_t = self.do_matrix_mult(proj_direction, plane_normal)
        v_dot_n = self.do_dot_product(proj_direction, plane_normal)

        A = self.do_matrix_sub(identity, self.do_scalar_mult(1/v_dot_n, vn_t))

        p = self.do_scalar_mult(self.do_dot_product(plane_point, plane_normal)/v_dot_n, proj_direction)

        output_file = open(filename, 'w', encoding="utf-8")

        new_line_counter = 1
        for point in self.points:
            if new_line_counter == 4:
                output_file.write(f'\n ')
                new_line_counter = 1
            new_line_counter += 1
            output_file.write(" ".join([str(x) for x in self.do_matrix_add(self.do_matrix_mult(A, point), p)]))
            output_file.write(" ")

        output_file.close()

    # This function uses equation (10.10) from the textbook to perform a perspective projection
    def do_perspective_projection(self, filename):
        plane_point = self.plane[0]
        plane_normal = self.do_normalize(self.plane[1])
        proj_direction = self.direction

        output_file = open(filename, 'w', encoding="utf-8")

        new_line_counter = 1
        for point in self.points:
            if new_line_counter == 4:
                output_file.write(f'\n ')
                new_line_counter = 1
            new_line_counter += 1

            q_dot_n = self.do_dot_product(plane_point, plane_normal)
            x_dot_n = self.do_dot_product(point, plane_normal)

            output_file.write(" ".join([str(x) for x in self.do_scalar_mult(q_dot_n/x_dot_n, point)]))
            output_file.write(" ")

        output_file.close()

    # This function calculates the distance between a point and a plane by generating the point normal form of
    # the plane and then plugging the point into the plane equation.
    def do_calculate_distance(self, filename):

        output_file = open(filename, 'w', encoding="utf-8")

        for problem in self.planes_and_points:
            plane_point = problem[0]
            plane_normal = self.do_normalize(problem[1])
            point = problem[2]

            c = (self.do_dot_product(plane_point, plane_normal)) * (-1)

            distance = c + self.do_dot_product(plane_normal, point)

            output_file.write(f'{distance}\n')

        output_file.close()

    # This function calculates the intersection point of a line and a bounded triangular plane, or prints a message
    # indicating that there is no intersection. It first calculates the intersection regardless of the triangular
    # bounds and then checks if the point is within those bounds.
    def do_intersection(self, filename):

        output_file = open(filename, 'w', encoding="utf-8")

        line_point = self.line[0]
        line_v = self.do_matrix_sub(self.line[1], self.line[0])

        # Loops for each triangle
        for triangle in self.triangles:
            p1 = triangle[0]
            p2 = triangle[1]
            p3 = triangle[2]

            # Generate vectors that define the triangular plane
            r1 = self.do_matrix_sub(p2, p1)
            r2 = self.do_matrix_sub(p3, p1)
            line_neg_v = self.do_scalar_mult(-1, line_v)

            # Define matrix A for the system
            matrix_A = [r1, r2, line_neg_v]
            matrix_A_transpose_tuples = list(zip(*matrix_A))
            matrix_A = [list(column) for column in matrix_A_transpose_tuples]

            # Solve with LU decomposition
            u1, u2, t = self.do_LU_decomposition(matrix_A, self.do_matrix_sub(line_point, p1))

            intersection = self.do_matrix_add(line_point, self.do_scalar_mult(t, line_v))

            if (0 <= u1 <= 1) & (0 <= u1 <= 1) & (u1 + u2 <= 1):
                output_file.write(f'{" ".join([str(x) for x in intersection])}\n')
            else:
                output_file.write("Does not intersect.\n")

        output_file.close()

    # This function employs the power method to find the eigenvector of a stochastic matrix that corresponds with
    # the eigenvalue 1. It prints the eigenvector as well as a list containing the page indexes sorted according to
    # their rank.
    def do_page_rank(self, filename):
        eigen_v = [1 for row in self.stochastic]
        j_inf_norm = eigen_v.index(max(eigen_v))
        y_vector = eigen_v
        k = 2
        eigen_lamd = [0, 0]
        tolerance = 0.01
        end_loop = False

        while not end_loop:

            y_vector = self.do_matrix_mult(self.stochastic, eigen_v)

            eigen_lamd.append(y_vector[j_inf_norm])

            j_inf_norm = y_vector.index(max(y_vector))

            eigen_v = self.do_scalar_mult(1/y_vector[j_inf_norm], y_vector)

            if abs(eigen_lamd[k] - eigen_lamd[k - 1]) < tolerance:
                end_loop = True
            k += 1

        dominant_eigen_v = eigen_v.copy()
        eigen_v_sorted = eigen_v.copy()
        eigen_v_sorted.sort(reverse=True)

        rankings = []
        for i in range(len(eigen_v)):
            rankings.append(eigen_v.index(eigen_v_sorted[i]) + 1)
            eigen_v[eigen_v.index(eigen_v_sorted[i])] = "blank"

        output_file = open(filename, 'w', encoding="utf-8")
        output_file.write(f'{" ".join([str(round(x, 2)) for x in dominant_eigen_v])}\n')
        output_file.write(f'{" ".join([str(x) for x in rankings])}\n')

        output_file.close()

    # This function employs the LU decompostition algorithm to solve a system with a 3x3 matrix A
    # and returns the 3 components of the solution as a tuple
    def do_LU_decomposition(self, matrix_A, vector_b):
        matrix_L = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        matrix_U = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        # Dimensions of the matrix
        n = len(matrix_A)

        # The LU decomposition algorithm
        for k in range(n):

            k_sum = 0
            for x in range(k):
                k_sum -= (matrix_L[k][x] * matrix_U[x][k])

            matrix_U[k][k] = matrix_A[k][k] + k_sum

            for i in range(k + 1, n):

                i_sum = 0
                for y in range(k):
                    i_sum -= (matrix_L[i][y] * matrix_U[y][k])

                matrix_L[i][k] = (1 / matrix_U[k][k]) * (matrix_A[i][k] + i_sum)

            for j in range(k + 1, n):

                j_sum = 0
                for z in range(k):
                    j_sum -= matrix_L[k][z] * matrix_U[z][j]

                matrix_U[k][j] = matrix_A[k][j] + j_sum

        # Calculate Ly = u with forward substitution
        y1 = vector_b[0]
        y2 = vector_b[1] - (matrix_L[1][0] * y1)
        y3 = vector_b[2] - (matrix_L[2][0] * y1) - (matrix_L[2][1] * y2)

        # Calculate Uu = b with forward substitution
        u3 = y3/matrix_U[2][2]
        u2 = (y2 - matrix_U[1][2] * u3) / matrix_U[1][1]
        u1 = (y1 - matrix_U[0][1] * u2 - matrix_U[0][2] * u3) / matrix_U[0][0]

        return u1, u2, u3

    def do_matrix_mult(self, mat_a, mat_b):

        try:
            product = [[0 for i in range(len(mat_b[0]))] for j in range(len(mat_a))]

            for i in range(len(mat_a)):
                for j in range(len(mat_b[0])):
                    for k in range(len(mat_b)):
                        product[i][j] += mat_a[i][k] * mat_b[k][j]

        except TypeError:
            try:
                product = [[0 for i in range(len(mat_b))] for j in range(len(mat_a))]

                for i in range(len(mat_a)):
                    for j in range(len(mat_b)):
                        product[i][j] += mat_a[i] * mat_b[j]

            except TypeError:
                product = [0 for i in range(len(mat_b))]

                for i in range(len(mat_a)):
                    for j in range(len(mat_a[0])):
                        product[i] += mat_a[i][j] * mat_b[j]

        return product

    # N x N matrix subtraction
    def do_matrix_sub(self, mat_a, mat_b):

        try:
            difference = [[0 for i in range(len(mat_a))] for i in range(len(mat_a))]

            for i in range(len(mat_a)):
                for j in range(len(mat_a[0])):
                    difference[i][j] = mat_a[i][j] - mat_b[i][j]

        except TypeError:
            difference = [0 for i in range(len(mat_a))]

            for i in range(len(mat_a)):
                difference[i] = mat_a[i] - mat_b[i]

        return difference

    # N x N matrix addition
    def do_matrix_add(self, mat_a, mat_b):

        try:
            matrix_sum = [[0 for i in range(len(mat_a))] for i in range(len(mat_a))]

            for i in range(len(mat_a)):
                for j in range(len(mat_a[0])):
                    matrix_sum[i][j] = mat_a[i][j] + mat_b[i][j]

        except TypeError:
            matrix_sum = [0 for i in range(len(mat_a))]

            for i in range(len(mat_a)):
                matrix_sum[i] = mat_a[i] + mat_b[i]

        return matrix_sum

    # N x N matrix scalar multiplication
    def do_scalar_mult(self, scalar, matrix):
        product = matrix

        try:
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    product[i][j] = scalar * matrix[i][j]

        except TypeError:
            for i in range(len(matrix)):
                product[i] = scalar * matrix[i]

        return product

    # N dimensional Vector dot product
    def do_dot_product(self, vect_a, vect_b):
        dot_product = 0

        for i in range(len(vect_a)):
            dot_product += vect_a[i] * vect_b[i]

        return dot_product


planes_points_lines = Planes_Points_Lines()

planes_points_lines.get_input_projection("inputHW4_Part1_2.txt")
planes_points_lines.do_parallel_projection("parallel_projection_output.txt")
planes_points_lines.do_perspective_projection("perspective_projection_output.txt")

planes_points_lines.get_input_distance("inputHW4_Part1_2.txt")
planes_points_lines.do_calculate_distance("distance_output.txt")

planes_points_lines.get_input_intersection("inputHW4_Part1_2.txt")
planes_points_lines.do_intersection("intersection_output.txt")

planes_points_lines.get_input_stochastic("inputHW4_Part3.txt")
planes_points_lines.do_page_rank("page_rank_output.txt")

planes_points_lines.get_input_stochastic("test_stochastic_input.txt")
planes_points_lines.do_page_rank("test_page_rank_output.txt")

planes_points_lines.get_input_stochastic("test_stochastic_input2.txt")
planes_points_lines.do_page_rank("test_page_rank_output2.txt")
