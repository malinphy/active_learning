import numpy as np 

def least_confidence_sampling(probabilities):
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)
    max_probs = np.max(probabilities, axis=1)
    return 1 - max_probs

def margin_sampling(probabilities):
    sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
    return margins

def entropy_sampling(probabilities):
    epsilon = 1e-10
    probabilities = probabilities + epsilon
    entropies = -np.sum(probabilities * np.log(probabilities), axis=1)
    return entropies


def active_learning_indices(metrics, sampling_function , k=10):
    indices_confidence_pairs = []
    for i in metrics["test_index_probabilites_per_epoch"][0:k]:
        index = int(i['index'])
        confidence = sampling_function(i['probabilities']).reshape(1, -1)[0][0]
        indices_confidence_pairs.append([index, confidence])

    least_confident_indices = [i[0] for i in sorted(indices_confidence_pairs, key=lambda x: x[1])]
    return least_confident_indices