import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
    
def cd(name = None,s = None,labels = None,alpha = None,clique = None): 
    
    # CRITICALDIFFERNCE - plot a critical difference diagram
    
    #    CRITICALDIFFERENCE(S,LABELS) produces a critical difference diagram [1]
#    displaying the statistical significance (or otherwise) of a matrix of
#    scores, S, achieved by a set of machine learning algorithms.  Here
#    LABELS is a cell array of strings giving the name of each algorithm.
    
    #    CRITICALDIFFERENCE(S,LABELS,ALPHA) allows the threshold of statistical
#    significance to be specified (defailt value 0.1).
    
    #    CRITICALDIFFERENCE(S,LABELS,ALPHA,CLIQUE) allows the cliques of
#    statistically equivalent algorithms to be pre-specified, rather than
#    using the usual Nemenyi post-hoc tests.  Clique is an N by M logical
#    matrix where N is the number of cliques and M is the number of
#    algorithms.  The value in column m of row n indcates whether algorithm m
#    belongs to clique n.
    
    #    [1] Demsar, J., "Statistical comparisons of classifiers over multiple
#        datasets", Journal of Machine Learning Research, vol. 7, pp. 1-30,
#        2006.
    
    # Thanks to Gideon Dror for supplying the extended table of critical values.
    
    if len(varargin) < 3:
        alpha = 0.1
    
    # convert scores into ranks
    
    N,k = s.shape
    S,r = __builtint__.sorted(np.transpose(s))
    idx = k * np.transpose(np.matlib.repmat(np.arange(0,N - 1+1),k,1)) + np.transpose(r)
    R = np.matlib.repmat(np.arange(1,k+1),N,1)
    S = np.transpose(S)
    for i in np.arange(1,N+1).reshape(-1):
        for j in np.arange(1,k+1).reshape(-1):
            index = S(i,j) == S(i,:)
            R[i,index] = mean(R(i,index))
    
    r[idx] = R
    r = np.transpose(r)
    # compute critical difference
    
    if alpha == 0.01:
        qalpha = np.array([0.0,2.576,2.913,3.113,3.255,3.364,3.452,3.526,3.59,3.646,3.696,3.741,3.781,3.818,3.853,3.884,3.914,3.941,3.967,3.992,4.015,4.037,4.057,4.077,4.096,4.114,4.132,4.148,4.164,4.179,4.194,4.208,4.222,4.236,4.249,4.261,4.273,4.285,4.296,4.307,4.318,4.329,4.339,4.349,4.359,4.368,4.378,4.387,4.395,4.404,4.412,4.42,4.428,4.435,4.442,4.449,4.456])
    else:
        if alpha == 0.05:
            qalpha = np.array([0.0,1.96,2.344,2.569,2.728,2.85,2.948,3.031,3.102,3.164,3.219,3.268,3.313,3.354,3.391,3.426,3.458,3.489,3.517,3.544,3.569,3.593,3.616,3.637,3.658,3.678,3.696,3.714,3.732,3.749,3.765,3.78,3.795,3.81,3.824,3.837,3.85,3.863,3.876,3.888,3.899,3.911,3.922,3.933,3.943,3.954,3.964,3.973,3.983,3.992,4.001,4.009,4.017,4.025,4.032,4.04,4.046])
        else:
            if alpha == 0.1:
                qalpha = np.array([0.0,1.645,2.052,2.291,2.46,2.589,2.693,2.78,2.855,2.92,2.978,3.03,3.077,3.12,3.159,3.196,3.23,3.261,3.291,3.319,3.346,3.371,3.394,3.417,3.439,3.459,3.479,3.498,3.516,3.533,3.55,3.567,3.582,3.597,3.612,3.626,3.64,3.653,3.666,3.679,3.691,3.703,3.714,3.726,3.737,3.747,3.758,3.768,3.778,3.788,3.797,3.806,3.814,3.823,3.831,3.838,3.846])
            else:
                raise Exception('alpha must be 0.01, 0.05 or 0.1')
    
    cd = qalpha(k) * np.sqrt(k * (k + 1) / (6 * N))
    f = plt.figure('Name',name,'visible','off')
    set(f,'Units','normalized')
    set(f,'Position',np.array([0,0,0.7,0.5]))
    clf
    plt.axis('off')
    plt.axis(np.array([- 0.5,1.5,0,140]))
    plt.axis('xy')
    tics = np.matlib.repmat((np.arange(0,(k - 1)+1)) / (k - 1),3,1)
    line(tics,np.matlib.repmat(np.array([100,105,100]),1,k),'LineWidth',2,'Color','k')
    tics = np.matlib.repmat(((np.arange(0,(k - 2)+1)) / (k - 1)) + 0.5 / (k - 1),3,1)
    line(tics,np.matlib.repmat(np.array([100,102.5,100]),1,k - 1),'LineWidth',1,'Color','k')
    #line([0 0 0 cd/(k-1) cd/(k-1) cd/(k-1)], [127 123 125 125 123 127], 'LineWidth', 1, 'Color', 'k');
#h = text(0.5*cd/(k-1), 130, 'CD', 'FontSize', 12, 'HorizontalAlignment', 'center');
    
    for i in np.arange(1,k+1).reshape(-1):
        text((i - 1) / (k - 1),110,num2str(k - i + 1),'FontSize',18,'HorizontalAlignment','center')
    
    # compute average ranks
    
    r = mean(r)
    r,idx = __builtint__.sorted(r)
    # compute statistically similar cliques
    
    if len(varargin) < 4:
        clique = np.matlib.repmat(r,k,1) - np.matlib.repmat(np.transpose(r),1,k)
        clique[clique < 0] = realmax
        clique = clique < cd
        for i in np.arange(k,2+- 1,- 1).reshape(-1):
            if np.all(clique(i - 1,clique(i,:)) == clique(i,clique(i,:))):
                clique[i,:] = 0
        n = np.sum(clique, 2-1)
        clique = clique(n > 1,:)
    else:
        if (len(clique) > 0):
            clique = clique(:,idx) > 0
    
    n = clique.shape[1-1]
    # labels displayed on the right
    
    for i in np.arange(1,np.ceil(k / 2)+1).reshape(-1):
        line(np.array([(k - r(i)) / (k - 1),(k - r(i)) / (k - 1),1.2]),np.array([100,100 - 5 * (n + 1) - 10 * i,100 - 5 * (n + 1) - 10 * i]),'Color','k')
        h = text(1.2,100 - 5 * (n + 1) - 10 * i + 5,num2str(r(i)),'FontSize',24,'HorizontalAlignment','right')
        text(1.25,100 - 5 * (n + 1) - 10 * i + 4,labels[idx(i)],'FontSize',28,'VerticalAlignment','middle','HorizontalAlignment','left')
    
    # labels displayed on the left
    
    for i in np.arange(np.ceil(k / 2) + 1,k+1).reshape(-1):
        line(np.array([(k - r(i)) / (k - 1),(k - r(i)) / (k - 1),- 0.2]),np.array([100,100 - 5 * (n + 1) - 10 * (k - i + 1),100 - 5 * (n + 1) - 10 * (k - i + 1)]),'Color','k')
        text(- 0.2,100 - 5 * (n + 1) - 10 * (k - i + 1) + 5,num2str(r(i)),'FontSize',24,'HorizontalAlignment','left')
        text(- 0.25,100 - 5 * (n + 1) - 10 * (k - i + 1) + 4,labels[idx(i)],'FontSize',28,'VerticalAlignment','middle','HorizontalAlignment','right')
    
    # group cliques of statistically similar classifiers
    
    for i in np.arange(1,clique.shape[1-1]+1).reshape(-1):
        R = r(clique(i,:))
        line(np.array([((k - np.amin(R)) / (k - 1)) + 0.015((k - np.amax(R)) / (k - 1)) - 0.015]),np.array([100 - 5 * i,100 - 5 * i]),'LineWidth',6,'Color','k')
    
    set(f,'CreateFcn','set(gcbo,'Visible','on')')
    saveas(f,name)
    # all done...
    return cd,f