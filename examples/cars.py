import pandas as pd

from pypbl.priors import Normal, Exponential
from pypbl.elicitation import BayesPreference

data = pd.read_csv('data/mtcars.csv')
print(data)

# set index of the data frame to be the item names
data.set_index('model', inplace=True)

p = BayesPreference(data=data)
p.set_priors([
    Exponential(1),  # MPG - high miles per gallon is preferred
    Normal(),  # number of cylinders
    Normal(),  # displacement
    Exponential(2),  # horsepower - high horsepower is preferred
    Normal(),  # real axle ratio
    Normal(),  # weight
    Exponential(-3),  # quarter mile time - high acceleration is preferred
    Normal(),  # engine type
    Normal(),  # transmission type
    Normal(),  # number of gears
    Normal()  # number of carburetors
])

# add some preferences and infer the weights for each parameters
p.add_strict_preference('Pontiac Firebird', 'Fiat 128')
p.add_strict_preference('Mazda RX4', 'Mazda RX4 Wag')
p.add_indifferent_preference('Merc 280', 'Merc 280C')
p.infer_weights(method='mean')

print('\ninferred weights')
for a, b in zip(data.columns.values.tolist(), p.weights.tolist()):
    print('{}: {}'.format(a, b))

# rank all the items and highlight the top five
print('\ntop 5 cars')
print(p.rank().head(5))

# suggest a new item to compare against the highest ranked solution - this may take some time to compute
print('\nsuggested pair to request new preference')
print(p.suggest())
