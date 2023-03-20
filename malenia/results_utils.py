

## Class to declare metrics to be used in the evaluation of the results
#
class Metric:
    def __init__(self, metric, **kwargs):
        self.metric = metric
        self.kwargs = kwargs

    def compute(self, y_true, y_pred):
        return self.metric(y_true, y_pred, **self.kwargs)

## Functions to compute cliques in CDD diagrams
#
def findCliques(same):
    cliques = []
    prevEndOfClique = 0
    
    for i in range(len(same)):
        clique = [i]
        growClique(same, clique)

        if len(clique) > 1:
            endOfClique = clique[-1]
            if endOfClique > prevEndOfClique:
                cliques.append(clique)
                prevEndOfClique = endOfClique
    
    finalCliques = [[False]*len(same) for _ in range(len(cliques))]
    for i in range(len(cliques)):
        for j in range(len(cliques[i])):
            finalCliques[i][cliques[i][j]] = True
    
    return finalCliques
#
#
def growClique(same, clique):
    prevVal = clique[-1]
    if prevVal == len(same)-1:
        return

    cliqueStart = clique[0]
    nextVal = prevVal+1

    for col in range(cliqueStart, nextVal):
        if not same[nextVal][col]:
            return
    
    clique.append(nextVal)
    growClique(same, clique)
#
#
def testNewCliques():
    same = [
        [True,  True,  True,  False, False, False],
        [True,  True,  True,  True,  False, False],
        [True,  True,  True,  False, True,  True],
        [False, True,  False, True,  True,  True],
        [False, False, True,  True,  True,  True],
        [False, False, True,  True,  True,  True],
    ]
    
    noDifference = same