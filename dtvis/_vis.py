import numpy as np

def print_tree_traversal(dt, feature_names=None, indent=4):
    left = dt.tree_.children_left
    right = dt.tree_.children_right
    threshold = dt.tree_.threshold
    if feature_names:
        features = [feature_names[f] if f >= 0 else None for f in dt.tree_.feature]
    else:
        features = ['x{}'.format(f) if f >= 0 else None for f in dt.tree_.feature]
    size = np.asarray([freq.sum() for freq in dt.tree_.value], dtype=np.int)
    prob = np.asarray([freq/freq.sum() for freq in dt.tree_.value])
    labels = np.asarray([prob_.argmax() for prob_ in prob])
    _print_tree_traversal(left, right, features, threshold, labels, size, prob, indent)

def _print_tree_traversal(left, right, features, threshold, labels, size, prob, indent=4):

    def is_leaf(i):
        return features[i] is None

    def print_status(i, depth, equation):
        message = '{} ({}). label={} n_samples={}, prob=({})'.format(
            '|--- ' * depth, # indention
            equation,               # equation
            labels[i],              # label
            size[i],                # n samples
            ', '.join(['%.3f' % float(p) for p in prob[i][0]])) # prob
        print(message, flush=True)

    def make_equation(idx, depth):
        # (child idx, depth, equation)
        equation = [
            (right[idx], depth, '{} > {}'.format(features[idx], '%.3f'%threshold[idx])),
            (left[idx], depth, '{} < {}'.format(features[idx], '%.3f'%threshold[idx]))
        ]
        return equation

    # initialize
    stack = make_equation(idx=0, depth=1)

    # print root
    print('|--- Root n_samples={}, prob=({})'.format(
        size[0], ', '.join(['%.3f' % float(p) for p in prob[0][0]])))

    # while stack is not empty
    while stack:
        idx, depth, equation = stack.pop()
        # if node is leaf print status        
        if is_leaf(idx):
            print_status(idx, depth, equation)
        # else print status and add children (left, right) order
        else:
            print_status(idx, depth, equation)
            stack += make_equation(idx, depth+1)