import json

from ace import run_ace, hider

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR


import pandas, numpy
import random, math

random.seed(0)


def get_learning_data_regression_tree():
    external = ['price', 'sales', 'workers', 'sold', 'profit', 'world_unemployment_rate', 'world_salary', 'world_sales', 'world_sold']#, 'raw', 'capital']
    control_parameters = ['salary', 'plan']

    history = pandas.read_csv('D:/pet projects/test_ace/firm_oligopoly_changes_history.csv', header = 0)

    learning_data = {}

    for parameter in control_parameters:
       learning_data['regression_' + parameter] =  DecisionTreeRegressor()
       learning_data['regression_' + parameter].fit(history[external], history[[parameter]])

    for parameter in ['world_unemployment_rate', 'world_salary', 'world_sales', 'world_sold']:
         learning_data['regression_' + parameter] = DecisionTreeRegressor()
         learning_data['regression_' + parameter].fit(history[['price', 'sales', 'workers', 'sold', 'profit']], history[[parameter]])
    return learning_data

def get_learning_data_neural_network():
    external = ['salary', 'plan', 'price', 'world_salary', 'world_price', 'world_sold', 'world_sales']#, 'raw', 'capital']
    control_parameters = ['salary', 'plan']

    history = pandas.read_csv('D:/pet projects/test_ace/firm_changes_history.csv', header = 0)

    history['has_profit'] = numpy.where(history[['profit']] > 0, 1, 0)

    learning_data = {}


    learning_data['neural_network'] = MLPClassifier(hidden_layer_sizes=(10, ))
    learning_data['neural_network'].fit(history[external], history[['has_profit']])

    for parameter in ['world_salary', 'world_sales', 'world_sold', 'world_price']:
       learning_data['neural_network_' + parameter] =  MLPRegressor(hidden_layer_sizes=(10, ))
       learning_data['neural_network_' + parameter].fit(history[['salary', 'plan', 'price']], history[[parameter]])

    return learning_data
   
def get_learning_data_classification_tree():
    external = ['salary', 'plan', 'price', 'world_salary', 'world_price', 'world_sold', 'world_sales']#, 'raw', 'capital']
    control_parameters = ['salary', 'plan']

    history = pandas.read_csv('D:/pet projects/test_ace/firm_changes_history.csv', header = 0)

    history['has_profit'] = numpy.where(history[['profit']] > 0, 1, 0)

    learning_data = {}


    learning_data['classification_tree'] = DecisionTreeClassifier()
    learning_data['classification_tree'].fit(history[external], history[['has_profit']])

    for parameter in ['world_salary', 'world_sales', 'world_sold', 'world_price']:
       learning_data['regression_tree_' + parameter] =  DecisionTreeRegressor()
       learning_data['regression_tree_' + parameter].fit(history[['salary', 'plan', 'price']], history[[parameter]])

    return learning_data

def get_learning_data_svm():
    external = ['salary', 'plan', 'price', 'world_salary', 'world_price', 'world_sold', 'world_sales']#, 'raw', 'capital']
    control_parameters = ['salary', 'plan']

    history = pandas.read_csv('D:/pet projects/test_ace/firm_changes_history.csv', header = 0)

    history['has_profit'] = numpy.where(history[['profit']] > 0, 1, 0)

    learning_data = {}

    learning_data['svm'] = SVC()
    learning_data['svm'].fit(history[external], history[['has_profit']])

    for parameter in ['world_salary', 'world_sales', 'world_sold', 'world_price']:
       learning_data['svm_' + parameter] =  SVR()
       learning_data['svm_' + parameter].fit(history[['salary', 'plan', 'price']], history[[parameter]])

    return learning_data

def get_learning_data_hider():
    external = ['price', 'sales', 'workers', 'plan', 'profit', 'world_unemployment_rate', 'world_salary', 'world_sales', 'world_sold']#, 'raw', 'capital']
    control_parameters = ['salary', 'sold']

    history = pandas.read_csv('D:/pet projects/test_ace/firm_oligopoly_changes_history.csv', header = 0)

#    train = list(numpy.random.choice(history.shape[0], size = math.floor(history.shape[0] * 0.8), replace=False))
#    test = list(set(range(history.shape[0])) - set(train))

    learning_data = {}

    for parameter in control_parameters:
        learning_data['hider_' + parameter] = pandas.read_csv('D:/pet projects/test_ace/rules_' + parameter + '.csv', header = 0).values.tolist()
        #else:
        #learning_data['hider_' + parameter] =  hider.hider(Sample([row for row in history.iloc[train][external + [parameter]].values.tolist()]))
        #pandas.DataFrame.from_records(learning_data['hider_' + parameter], columns = [position + '_' + variable for variable in external + [parameter] for position in ['lower', 'upper']]).to_csv('D:/pet projects/test_ace/test_sample_rules_' + parameter + '.csv')
        #learning_data['correct_rules_' + parameter] = hider.correct_rules(learning_data['hider_' + parameter], history.iloc[test][external + [parameter]].values.tolist())

#    for parameter in control_parameters:
#        print('Correct rules for ' + parameter + ': ' + str(learning_data['correct_rules_' + parameter]))



    for parameter in ['world_unemployment_rate', 'world_salary', 'world_sales', 'world_sold']:
        learning_data['hider_' + parameter] = pandas.read_csv('D:/pet projects/test_ace/rules_' + parameter + '.csv',
                                                              header=0).values.tolist()
        #learning_data['hider_' + parameter] =  hider.hider(Sample([row for row in history[['price', 'sales', 'workers', 'sold', 'profit']].values.tolist()]))
        #pandas.DataFrame.from_records(learning_data['hider_' + parameter],
        #columns = [position + '_' + variable for variable in ['price', 'sales', 'workers', 'sold', 'profit'] for position in ['lower', 'upper']]).to_csv('D:/pet projects/test_ace/rules_' + parameter + '.csv')

    return learning_data

#learning_data = get_learning_data_regression_tree()#get_learning_data_hider()

#learning_data.update(get_learning_data_classification_tree())
#learning_data.update(get_learning_data_neural_network())
#learning_data.update(get_learning_data_regression_tree())
#learning_data.update(get_learning_data_svm())

learning_data = None

#run_ace.run_ace_from_file("model_config.json", "run_config.json")
#for i in range (0, 50, 50):

i = 0

for tax_delta in [0.025, 0.05, 0.1, 0.2]:
    tax = 0
    while tax <1:
        random.seed(0)
        run_ace.run_ace_from_file("model_config_intuitive_firm.json", "run_config_intuitive_firm.json", tax, learning_data)
        tax += tax_delta
        print(str(i) + ' ' + str(0) + ' ' + str(tax))
        i += 1
#run_ace.run_ace_from_file("model_config_intuitive_firm_all_types.json", "run_config_intuitive_firm_all_types.json")
#run_ace.run_ace_from_file("model_config_qlearning_firm_all_types.json", "run_config_qlearning_firm_all_types.json")
#run_ace.run_ace_from_file("model_config_qlearning.json", "run_config_qlearning.json")
#run_ace.run_ace_from_file("model_config_nonconscious.json", "run_config_nonconscious.json")
#run_ace.run_ace_from_file("model_config_budget_firm.json", "run_config_budget_firm.json")
#run_ace.run_ace_from_file("model_config_extrapolation.json", "run_config_extrapolation.json")
#run_ace.run_ace_from_file("model_config_oligopoly.json", "run_config_oligopoly.json")
#run_ace.run_ace_from_file("model_config_rational.json", "run_config_rational.json")
#run_ace.run_ace_from_file("model_config_random.json", "run_config_random.json")
#run_ace.run_ace_from_file("model_config_moses_firm.json", "run_config_moses_firm.json")
#run_ace.run_ace_from_file("model_config_rule_firm.json", "run_config_rule_firm.json")