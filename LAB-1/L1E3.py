import numpy as np

arr = np.arange(1,13)

reshaped = arr.reshape(3,4)

print("Original Array:", arr)
print("Shape:", arr.shape)

print("\nReshaped Array:\n", reshaped)
print("Shape:", reshaped.shape)
