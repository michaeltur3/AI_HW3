

def sfs(x, y, k, clf, score):
    """
    :param x: feature set to be trained using clf. list of lists.
    :param y: labels corresponding to x. list.
    :param k: number of features to select. int
    :param clf: classifier to be trained on the feature subset.
    :param score: utility function for the algorithm, that receives clf, feature subset and labeles, returns a score. 
    :return: list of chosen feature indexes
    """
    V = set()
    num_of_features = len(x[0])
    N = set(list(range(num_of_features)))
    for i in range(k):
        max_score = 0
        best_v = None
        for v in N:
            V.add(v)
            V_list = list(V)
            cur_score = score(clf, V_list, x, y)
            if cur_score > max_score:
                max_score = cur_score
                best_v = v
            V.remove(v)
        V.add(best_v)
        N.remove(best_v)
    return list(V)
