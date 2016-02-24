'''Simple script to parse the master net's training log, to compute and save
the confusion matrix.

The file in PATH should only contain the matrix dumped on report.html by the
solver.
'''

import numpy as np
import matplotlib.pyplot as plt
import string

def int_filter(s):
	'''Only keep strings containing integers.'''
	if not s:
		return True
	return s.isdigit() or s.split(']')[0].isdigit()

PATH = 'errors.txt'

# Parse file to get confusion matrix
f = open(PATH, 'rb')
errors = []
for line in f:
	numbers = filter(int_filter, line.split(" "))
	numbers = map(lambda x: int(x) if x.isdigit() else int(x[0]), numbers)
	errors.extend(numbers)
f.close()

# Reshape and get rates
errors = np.array(errors).reshape((100, 100))
errors = errors / (np.sum(errors, axis=0) + 1e-7)

# Plot confusion matrix
fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(errors), cmap=plt.cm.jet, 
                interpolation='nearest')
cb = fig.colorbar(res)
plt.xlabel('Predictions')
plt.ylabel('Ground Truth')
plt.title('Rates')
plt.savefig('matrix.png', format='png')
plt.close()
