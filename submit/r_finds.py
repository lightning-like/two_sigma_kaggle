import pickle
import pandas as pd
import matplotlib.pyplot as plt

a = pickle.load(open('p.pkl','rb'))

l = []
r = abs(a).groupby('time').sum()

r = r['returnsOpenNextMktres10']

print(r.std())
print(r.mean()/r.std())

r[r>r.quantile()] = r[r< r.quantile()] * 0.5

print(r.std())
print(r.mean()/r.std())


r[r>r.quantile()] = r[r< r.quantile()] * 0.5

print(r.std())
print(r.mean()/r.std())

r.hist()
plt.show()