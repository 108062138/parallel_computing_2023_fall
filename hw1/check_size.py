import struct
import math
# Define the file path
file_path = '/home/pp23/pp23s80/HW/hw1/testcases/04.in'

# Open the binary file in binary read mode
with open(file_path, 'rb') as file:
    data = file.read()

# Initialize an empty list to store the extracted floating-point numbers
float_numbers = []

# Iterate through the data in 4-byte increments (32 bits)
for i in range(0, len(data), 4):
    # Extract each 4-byte chunk and convert it to a 32-bit float
    float_value = struct.unpack('f', data[i:i+4])[0]
    float_numbers.append(float_value)

# Print the extracted floating-point numbers
print("Extracted Floating-Point Numbers:")
for num in float_numbers:
    print(num)

# Calculate and print the number of floating-point numbers in the file
num_floats = len(float_numbers)
print(f"Number of Floating-Point Numbers in the File: {num_floats}")

# let size be the process we can use. find out their offset to partition n
size = 4

# case1: assume all process is utilized and no root
each_hold = math.ceil(num_floats/size)
if num_floats % each_hold==0: last_hold = each_hold
else: last_hold = num_floats % each_hold
print('size',size, 'each_hold', each_hold, 'last_hold', last_hold)
for rank in range(0, size):
    if rank!=size-1:
        print('at rank', rank, ',[',rank*each_hold, ',',rank*each_hold + each_hold -1,']')
    else:
        print('at rank', rank, ',[',rank*each_hold, ',',rank*each_hold + last_hold -1,']')

# case2: use root
each_hold = math.ceil(num_floats/(size-1))
if num_floats % each_hold==0: last_hold = each_hold
else: last_hold = num_floats % each_hold
print('size',size, 'each_hold', each_hold, 'last_hold', last_hold)
for rank in range(0, size):
    if rank<size-1 and rank>0:
        print('at rank', rank, ',[',(rank-1)*each_hold, ',',rank*each_hold -1,']')
    elif rank==size-1:
        print('at rank', rank, ',[',(rank-1)*each_hold, ',',(rank-1)*each_hold + last_hold -1,']')
    else:
        print('at rank 0, be root and do nothing')