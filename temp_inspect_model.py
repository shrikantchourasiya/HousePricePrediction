import pickle
pipe = pickle.load(open('Nofeature.pkl', 'rb'))
print('type:', type(pipe))
print('has named_steps:', hasattr(pipe, 'named_steps'))
if hasattr(pipe, 'named_steps'):
    print('named_steps keys:', list(pipe.named_steps.keys()))
print('has steps:', hasattr(pipe, 'steps'))
print('steps:', getattr(pipe, 'steps', None))
print('has transformers_:', hasattr(pipe, 'transformers_'))
if hasattr(pipe, 'transformers_'):
    print('transformers_:', pipe.transformers_)
for attr in ['_name_to_fitted_passthrough', 'transformers_', 'transformers', 'fit_transform', 'predict']:
    print(attr, hasattr(pipe, attr))

try:
    import sklearn
    print('sklearn', sklearn.__version__)
except Exception as e:
    print('sklearn import error', e)
