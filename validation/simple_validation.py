import numpy as np
import pandas as pd

import matplotlib.pylab as plt

from pypbl.elicitation import BayesPreference
from pypbl.priors import Normal


def calculate_error(y, y_pred):
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    error = np.arccos(np.dot(y, y_pred) / (np.linalg.norm(y) * np.linalg.norm(y_pred)))
    return error


# fix seed
np.random.seed(seed=240786)

# set up simple data set
n_items = 100
item_names = ['item {}'.format(i) for i in range(n_items)]

n_features = 3
feature_names = ['feature {}'.format(i) for i in range(n_features)]
features = np.random.rand(100, 3)
data = pd.DataFrame(features, columns=feature_names, index=item_names)

known_weights = (5, -1, 2)

print(data)

n_preferences = 30
n_repeats = 30

random_method = []
for test in range(n_repeats):
    # RANDOM METHOD
    print('RANDOM METHOD: {} of {}'.format(test, n_repeats))
    random_model = BayesPreference(data=data)
    random_model.set_priors([Normal() for i in range(n_features)])
    random_method_weight_error = []
    for i in range(n_preferences):
        if i == 0:
            suggested_pair = ['item 0', 'item 1']
        else:
            suggested_pair = random_model.suggest_new_pair(method='random')

        a_utility = sum([x * w for x, w in zip(data.loc[suggested_pair[0], :].values, known_weights)])
        b_utility = sum([x * w for x, w in zip(data.loc[suggested_pair[1], :].values, known_weights)])
        if a_utility > b_utility:
            random_model.add_strict_preference(suggested_pair[0], suggested_pair[1])
        elif b_utility > a_utility:
            random_model.add_strict_preference(suggested_pair[1], suggested_pair[0])
        else:
            random_model.add_indifferent_preference(suggested_pair[0], suggested_pair[1])

        predicted_weights = random_model.infer_weights()
        random_method_weight_error.append(calculate_error(known_weights, predicted_weights))

    random_method.append(random_method_weight_error)
    del random_model

random_with_best_method = []
for test in range(n_repeats):
    # RANDOM WITH BEST METHOD
    print('RANDOM WITH BEST METHOD: {} of {}'.format(test, n_repeats))
    random_with_best_model = BayesPreference(data=data)
    random_with_best_model.set_priors([Normal() for i in range(n_features)])
    random_with_best_method_weight_error = []
    for i in range(n_preferences):
        if i == 0:
            suggested_pair = ['item 0', 'item 1']
        else:
            suggested_pair = random_with_best_model.suggest(method='random')

        a_utility = sum([x * w for x, w in zip(data.loc[suggested_pair[0], :].values, known_weights)])
        b_utility = sum([x * w for x, w in zip(data.loc[suggested_pair[1], :].values, known_weights)])
        if a_utility > b_utility:
            random_with_best_model.add_strict_preference(suggested_pair[0], suggested_pair[1])
        elif b_utility > a_utility:
            random_with_best_model.add_strict_preference(suggested_pair[1], suggested_pair[0])
        else:
            random_with_best_model.add_indifferent_preference(suggested_pair[0], suggested_pair[1])

        predicted_weights = random_with_best_model.infer_weights()
        random_with_best_method_weight_error.append(calculate_error(known_weights, predicted_weights))

    random_with_best_method.append(random_with_best_method_weight_error)
    del random_with_best_model

uncertain_with_best_method = []
for test in range(n_repeats):
    # UNCERTAIN WITH BEST METHOD
    print('UNCERTAIN WITH BEST METHOD: {} of {}'.format(test, n_repeats))
    uncertain_with_best_model = BayesPreference(data=data)
    uncertain_with_best_model.set_priors([Normal() for i in range(n_features)])
    uncertain_with_best_method_weight_error = []
    for i in range(n_preferences):
        if i == 0:
            suggested_pair = ['item 0', 'item 1']
        else:
            suggested_pair = uncertain_with_best_model.suggest(method='max_variance')

        a_utility = sum([x * w for x, w in zip(data.loc[suggested_pair[0], :].values, known_weights)])
        b_utility = sum([x * w for x, w in zip(data.loc[suggested_pair[1], :].values, known_weights)])
        if a_utility > b_utility:
            uncertain_with_best_model.add_strict_preference(suggested_pair[0], suggested_pair[1])
        elif b_utility > a_utility:
            uncertain_with_best_model.add_strict_preference(suggested_pair[1], suggested_pair[0])
        else:
            uncertain_with_best_model.add_indifferent_preference(suggested_pair[0], suggested_pair[1])

        predicted_weights = uncertain_with_best_model.infer_weights()
        uncertain_with_best_method_weight_error.append(calculate_error(known_weights, predicted_weights))

    uncertain_with_best_method.append(uncertain_with_best_method_weight_error)
    del uncertain_with_best_model

plt.figure()
plt.plot(range(1, n_preferences + 1), np.mean(random_method, axis=0), label='random suggestion')
plt.plot(range(1, n_preferences + 1), np.mean(random_with_best_method, axis=0),
         label='best utility with random suggestion')
plt.plot(range(1, n_preferences + 1), np.mean(uncertain_with_best_method, axis=0),
         label='best utility with most uncertain suggestion')
plt.xlabel('number of preferences provided')
plt.ylabel('error is predicted weights')
plt.legend()
plt.savefig('img/validation.png')

plt.figure()
plt.errorbar(range(1, n_preferences + 1), np.mean(random_method, axis=0),
             yerr=np.std(random_method, axis=0), label='random suggestion')
plt.errorbar(range(1, n_preferences + 1), np.mean(random_with_best_method, axis=0),
             yerr=np.std(random_with_best_method, axis=0), label='best utility with random suggestion')
plt.errorbar(range(1, n_preferences + 1), np.mean(uncertain_with_best_method, axis=0),
             yerr=np.std(uncertain_with_best_method, axis=0), label='best utility with most uncertain suggestion')
plt.xlabel('number of preferences provided')
plt.ylabel('error is predicted weights')
plt.legend()
plt.savefig('img/mean_validation.png')

plt.figure()
c1 = [0.122, 0.467, 0.706]
c2 = [1, 0.498, 0.55]
c3 = [0.173, 0.627, 0.173]
alpha = [0.25]
box1 = plt.boxplot(np.array(random_method), showcaps=False, showfliers=False)
for _, line_list in box1.items():
    for line in line_list:
        line.set_color(tuple(c1 + alpha))
plt.plot(range(1, n_preferences + 1), np.median(random_method, axis=0), color=tuple(c1), label='random suggestion')
box2 = plt.boxplot(np.array(random_with_best_method), showcaps=False, showfliers=False)
for _, line_list in box2.items():
    for line in line_list:
        line.set_color(tuple(c2 + alpha))
plt.plot(range(1, n_preferences + 1), np.median(random_with_best_method, axis=0), color=tuple(c2),
         label='best utility with random suggestion')
box3 = plt.boxplot(np.array(uncertain_with_best_method), bootstrap=1000, showcaps=False, showfliers=False)
for _, line_list in box3.items():
    for line in line_list:
        line.set_color(tuple(c3 + alpha))
plt.plot(range(1, n_preferences + 1), np.median(uncertain_with_best_method, axis=0), color=tuple(c3),
         label='best utility with most uncertain suggestion')

plt.xlabel('number of preferences provided')
plt.ylabel('error is predicted weights')
plt.legend()
plt.savefig('img/box_validation.png')
