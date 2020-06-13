from run_end_to_end import run_program_binary, run_program_multi

data_balancing_types_binary = ['over-sample-seals', 'sub-sample-background', 'SMOTE']
data_balancing_types_multi = ['over-sample-seals', 'sub-sample', 'SMOTE', 'mixed-balancing']

solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
models = ['LogisticRegression', 'SVC']

# select the model parameters and uncomment the function which has to be run
'''
run_program_multi(standardize=True,
                  data_balancing_type=data_balancing_types_multi[0],
                  model=models[0],
                  cross_validation=True,
                  solver=solvers[1])
'''

run_program_binary(standardize=True,
                   data_balancing_type=data_balancing_types_binary[0],
                   model=models[1],
                   cross_validation=True,
                   threshold=0.85,
                   solver=solvers[1])





