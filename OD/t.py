import numpy as np

a=np.array([[1,11],[2,22],[3,33],[4,44],[5,55],[6,66],[1,11],[2,22],[3,33],[4,44],[5,55],[6,66]])
# print(a)

b=np.concatenate([a[i*6:(i+1)*6,:] for i in range(2)],axis=1)
print(b)
print(b.shape)
print(np.reshape(b,newshape=[3,2,4]))
# a=np.reshape(a,newshape=[-1,2])
# print(a)