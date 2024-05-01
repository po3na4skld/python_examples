def get_num_islands(m, n, matrix):
    """
    Calculates the number of distinct islands in a given 2D grid.

    An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.
    The grid is represented as a matrix of integers, where 1 = land and 0 = water.

    Parameters:
    - m (int): A number of rows in a matrix
    - n (int): A number of cols in a matrix
    - matrix (list of list of str): A 2D grid where each element is either 1 or 0

    Returns:
    - int: The total number of distinct islands in the grid.
    """

    if not matrix or not matrix[0]:
        return 0

    visited = [[False] * n for _ in range(m)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    count = 0

    def dfs(row_idx, col_idx):
        """
        Performs a depth-first search to mark all parts of the current island as visited.

        Parameters:
        - row_idx (int): The row index of the starting cell.
        - col_idx (int): The column index of the starting cell.
        """

        stack = [(row_idx, col_idx)]
        while stack:
            x, y = stack.pop()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny] and matrix[nx][ny] == 1:
                    visited[nx][ny] = True
                    stack.append((nx, ny))

    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1 and not visited[i][j]:
                visited[i][j] = True
                dfs(i, j)
                count += 1

    return count


# Example usage:
grid1 = [[0, 1, 0], [0, 0, 0], [0, 1, 1]]
grid2 = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]
grid3 = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]]

print(get_num_islands(3, 3, grid1))  # Output: 2
print(get_num_islands(3, 4, grid2))  # Output: 3
print(get_num_islands(3, 4, grid3))  # Output: 2
