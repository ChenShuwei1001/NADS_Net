import numpy as np
if __name__ == '__main__':
    a = np.array([[[1,2,3],[2,4,6],[3,6,9]]])
    print(a.shape)
    print(a)
    a[:,[i for i in range(3)],:] = a[:,[2,0,1],:]
    print(a)
    print(a.shape)

