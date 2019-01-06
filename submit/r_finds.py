import pickle

import matplotlib.pyplot as plt

a = pickle.load(open('p.pkl', 'rb'))

l = []

a = abs(a)

r = a.groupby('time').sum()

r = r['returnsOpenNextMktres10']

print(r.std())
print(r.mean() / r.std())

a[a > a.quantile()] = a[a > a.quantile()] * 0.5

r = a.groupby('time').sum()

r = r['returnsOpenNextMktres10']
print(r.std())
print(r.mean() / r.std())

a[a > a.quantile()] = a[a > a.quantile()] * 0.5

r = a.groupby('time').sum()

r = r['returnsOpenNextMktres10']
print(r.std())
print(r.mean() / r.std())

r.hist()
plt.show()
