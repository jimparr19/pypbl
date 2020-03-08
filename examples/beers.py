import pandas as pd

from pypbl.priors import Normal
from pypbl.elicitation import BayesPreference

data = pd.read_csv('data/bdbeers.csv')
print(data)

# set index of the data frame to be the item names
data.set_index('name', inplace=True)

# drop some columns that are unlikely to influence preferences
data.drop(columns=['srm', 'target_fg'], inplace=True)

p = BayesPreference(data=data)
p.set_priors([
    Normal(),  # abv - alcohol strength
    Normal(),  # attenuation_level
    Normal(),  # ebc - beer colour
    Normal(),  # ibu - bitterness
    Normal(),  # n_hops - number of hops
    Normal(),  # n_malt - number of malts
    Normal(),  # ph - ph level
    Normal(),  # target_og - target gravity
])

# add some preferences and infer the weights for each parameters
p.add_strict_preference('Indie Pale Ale', 'Kingpin')
p.add_strict_preference('Dead Pony Club', 'Indie Pale Ale')
p.add_strict_preference('Dead Pony Club', 'Punk IPA 2010 - Current')
p.add_strict_preference('5am Saint', 'Dead Pony Club')
p.add_strict_preference('Hazy Jane', '5am Saint')
p.add_strict_preference('Hazy Jane', 'Quench Quake')
p.add_strict_preference('Hazy Jane', 'Zombie Cake')
p.infer_weights(method='mean')

print('\ninferred weights')
for a, b in zip(data.columns.values.tolist(), p.weights.tolist()):
    print('{}: {}'.format(a, b))

# rank all the items and highlight the top five
print('\ntop 5 beers')
print(p.rank().head(5))

# suggest a new item to compare against the highest ranked solution - this may take some time to compute
print('\nsuggested pair to request new preference')
print(p.suggest())
