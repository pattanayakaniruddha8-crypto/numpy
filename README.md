import numpy as np

# 1. Create a 5×5 matrix with values 1 to 25
matrix_5x5 = np.arange(1, 26).reshape(5, 5)
print("5x5 Matrix:\n", matrix_5x5)

# 2. Generate a 4×4 identity matrix
identity_4x4 = np.eye(4)
print("\nIdentity Matrix:\n", identity_4x4)

# 3. 1D array from 100 to 200 with step 10
arr_step = np.arange(100, 201, 10)
print("\nArray 100-200 step 10:\n", arr_step)

# 4. Random 3×3 matrix & determinant
rand_matrix = np.random.rand(3, 3)
det = np.linalg.det(rand_matrix)
print("\nRandom 3x3:\n", rand_matrix, "\nDeterminant:", det)

# 5. Array of 10 random integers between 1 and 100
rand_ints = np.random.randint(1, 101, 10)
print("\nRandom Integers:\n", rand_ints)

# 6. Reshape 1D array of size 12 into 3×4
arr_12 = np.arange(12)
reshaped = arr_12.reshape(3, 4)
print("\nReshaped 3x4:\n", reshaped)

# 7. Matrix multiplication
A = np.random.randint(1, 10, (3, 3))
B = np.random.randint(1, 10, (3, 3))
matmul = np.dot(A, B)
print("\nMatrix Multiplication:\n", matmul)

# 8. Eigenvalues and eigenvectors of 2×2
mat_2x2 = np.array([[4, 2], [1, 3]])
eigvals, eigvecs = np.linalg.eig(mat_2x2)
print("\nEigenvalues:", eigvals, "\nEigenvectors:\n", eigvecs)

# 9. 5×5 random matrix & diagonal
rand_5x5 = np.random.rand(5, 5)
print("\nDiagonal Elements:", np.diag(rand_5x5))

# 10. Normalize 1D array
arr_norm = np.random.randint(1, 50, 10)
norm = (arr_norm - arr_norm.min()) / (arr_norm.max() - arr_norm.min())
print("\nNormalized Array:\n", norm)

# 11. Sort array by row & column
arr_sort = np.random.randint(1, 50, (4, 4))
print("\nSorted by row:\n", np.sort(arr_sort, axis=1))
print("Sorted by col:\n", np.sort(arr_sort, axis=0))

# 12. Indices of max & min
arr_vals = np.random.randint(1, 100, 10)
print("\nArray:", arr_vals, "Max idx:", arr_vals.argmax(), "Min idx:", arr_vals.argmin())

# 13. Flatten with ravel() & flatten()
arr2D = np.arange(1, 10).reshape(3, 3)
print("\nRavel:", arr2D.ravel(), "Flatten:", arr2D.flatten())

# 14. Inverse of 3×3
mat_inv = np.random.rand(3, 3)
print("\nInverse:\n", np.linalg.inv(mat_inv))

# 15. Random permutation of 1–10
print("\nRandom Permutation:", np.random.permutation(np.arange(1, 11)))

# 16. Replace even numbers with -1
arr_replace = np.arange(21)
arr_replace[arr_replace % 2 == 0] = -1
print("\nReplace evens with -1:\n", arr_replace)

# 17. Dot product
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print("\nDot product:", np.dot(x, y))

# 18. Trace of 5×5 random matrix
mat_trace = np.random.rand(5, 5)
print("\nTrace:", np.trace(mat_trace))

# 19. Split 1D array into 3 parts
arr_split = np.arange(9)
print("\nSplit into 3:\n", np.split(arr_split, 3))

# 20. 3D array (3,3,3) mean across axis=0
arr_3d = np.random.rand(3, 3, 3)
print("\nMean across axis=0:\n", arr_3d.mean(axis=0))

# 21. Cumulative sum
arr_cumsum = np.array([1, 2, 3, 4])
print("\nCumulative Sum:", np.cumsum(arr_cumsum))

# 22. Extract upper triangular
mat_upper = np.random.randint(1, 10, (4, 4))
print("\nUpper Triangular:\n", np.triu(mat_upper))

# 23. 6×6 Checkerboard pattern
checker = np.indices((6, 6)).sum(axis=0) % 2
print("\nCheckerboard:\n", checker)

# 24. Element-wise sqrt
mat_sqrt = np.random.rand(3, 3)
print("\nSqrt:\n", np.sqrt(mat_sqrt))

# 25. Reverse 1D array (without slicing)
arr_rev = np.arange(20)
print("\nReversed:", np.flip(arr_rev))

# 26. Merge arrays vertically & horizontally
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print("\nVertical Stack:\n", np.vstack((a, b)))
print("Horizontal Stack:\n", np.hstack((a, b)))

# 27. Row-wise & col-wise sum
mat_sum = np.arange(1, 10).reshape(3, 3)
print("\nRow Sum:", mat_sum.sum(axis=1), "Col Sum:", mat_sum.sum(axis=0))

# 28. Replace NaN with mean
arr_nan = np.array([[1, np.nan, 3], [4, 5, np.nan]])
col_mean = np.nanmean(arr_nan, axis=0)
inds = np.where(np.isnan(arr_nan))
arr_nan[inds] = np.take(col_mean, inds[1])
print("\nReplace NaN:\n", arr_nan)

# 29. Cosine similarity
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print("\nCosine Similarity:", cos_sim)

# 30. Rotate 4×4 array by 90 degrees
mat_rot = np.arange(16).reshape(4, 4)
print("\nRotated 90°:\n", np.rot90(mat_rot))

# 31. Structured array
dtype = [('name', 'U10'), ('age', 'i4'), ('marks', 'f4')]
students = np.array([("John", 20, 85.5), ("Alice", 22, 91.2)], dtype=dtype)
print("\nStructured Array:\n", students)

# 32. Rank of matrix
mat_rank = np.random.rand(3, 3)
print("\nRank:", np.linalg.matrix_rank(mat_rank))

# 33. Normalize rows to unit length
mat_norm = np.random.rand(5, 5)
mat_norm /= np.linalg.norm(mat_norm, axis=1, keepdims=True)
print("\nRow-normalized:\n", mat_norm)

# 34. Check arrays equal
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])
print("\nArrays equal:", np.array_equal(arr1, arr2))

# 35. Histogram
data = np.random.randn(1000)
hist, bins = np.histogram(data, bins=10)
print("\nHistogram:", hist)

# 36. Broadcasting
arr2D = np.ones((3, 3))
arr1D = np.array([1, 2, 3])
print("\nBroadcasting:\n", arr2D + arr1D)

# 37. Unique values & counts
arr_unique = np.array([1, 2, 2, 3, 4, 4, 4])
values, counts = np.unique(arr_unique, return_counts=True)
print("\nUnique values:", values, "Counts:", counts)

# 38. Pearson correlation
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
print("\nPearson Corr:\n", np.corrcoef(x, y))

# 39. Numerical gradient
arr_grad = np.array([1, 2, 4, 7, 11])
print("\nGradient:", np.gradient(arr_grad))

# 40. Singular Value Decomposition (SVD)
mat_svd = np.random.rand(3, 3)
U, S, V = np.linalg.svd(mat_svd)
print("\nSVD:\nU=\n", U, "\nS=", S, "\nV=\n", V)
