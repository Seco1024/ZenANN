import sys
sys.path.append('..')

from zenann import IndexBase

index = IndexBase(dim=64)
print(index.dimension) 

data = [[0.1] * 64 for _ in range(10)]
index.build(data)
index.train()
result = index.search([0.1] * 64, 3)

print("Indices:", result.indices)
print("Distances:", result.distances)
