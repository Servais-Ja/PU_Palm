import operator as op
import optunity.metrics
import semisup_metrics as ss
import numpy as np
from matplotlib import pyplot as plt
# convenience plot functions
def plot_proxy():
    p = plt.Rectangle((0, 0), 0, 0, color='blue', alpha=0.4)
    ax = plt.gca()
    ax.add_patch(p)
    return p
# convenience function for plot legend
def plot_proxy2():
    p = plt.Rectangle((0, 0), 0, 0, color='none',
                      edgecolor='red', hatch='xxx', alpha=0.8)
    ax = plt.gca()
    ax.add_patch(p)
    return p
def fix_plot_shape(fig):
    ax = fig.add_subplot(111, aspect='equal')
    axes = fig.gca()
    axes.set_xlim([0,1])
    axes.set_ylim([0,1])

def main_roc(pospath,unpath,title):
    #decision_values为打分值

    dv_lb_p=np.loadtxt(pospath,delimiter=',')
    dv_lb_u=np.loadtxt(unpath,delimiter=',')
    labels=np.hstack((np.ones(np.size(dv_lb_p,axis=0),dtype=bool),np.array([None]*np.size(dv_lb_u,axis=0))))
    decision_values=np.hstack((dv_lb_p[:,0],dv_lb_u[:,0]))

    # sort the labels in descending order of corresponding decision values降序排序
    sort_labels, sort_dv = zip(*sorted(zip(labels, decision_values),
                                                     key=op.itemgetter(1), reverse=True))#按decision_values进行降序排列，zip(*)解压

    ci_width = 0.95      # width of the confidence band on ECDF to be used
    use_bootstrap = False # use bootstrap to compute confidence band
    nboot = 2000         # number of bootstrap iterations to use, not used if use_bootstrap = False
    if use_bootstrap:
        cdf_bounds = ss.bootstrap_ecdf_bounds(labels, decision_values, nboot=nboot, ci_width=ci_width)
    else:
        cdf_bounds = ss.dkw_bounds(labels, decision_values, ci_width=ci_width)


    # first, compute contingency tables based on the point estimate betahat 根据β的点估计计算列联表
    # presorted = True is a computational shortcut
    # we can use this because we already sorted by decision values earlier
    tables = ss.compute_contingency_tables(labels=sort_labels, decision_values=sort_dv,
                                             reference_lb=cdf_bounds.lower,
                                             reference_ub=cdf_bounds.upper,
                                             beta=0.1456, presorted=True)

    _, roc_neg = optunity.metrics.roc_auc(labels, decision_values, return_curve=True)
    roc_bounds_function = lambda tables: ss._lb_ub(lower=ss.roc_from_cts(tables.lower),
                                      upper=ss.roc_from_cts(tables.upper))
    roc_bounds = roc_bounds_function(tables)

    xs = [float(x) / 100 for x in range(101)]
    roc_up = ss.zoh(*zip(*roc_bounds.upper))
    roc_lo = ss.zoh(*zip(*roc_bounds.lower))
    fig = plt.figure(2)
    fix_plot_shape(fig)
    plt.fill_between(xs, list(map(roc_lo, xs)), list(map(roc_up, xs)), color='orange', alpha=0.4)
    plt.plot(*zip(*roc_neg), color='red')#, linestyle='dashed'
    #plot_proxy()
    #plot_proxy2()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    #plt.legend(['beta=0','expected region via betahat'],loc="upper left", bbox_to_anchor=(1,1))
    plt.title(title)
    plt.show()

if __name__=='__main__':
    main_roc('../SVMscores/test_scores_p3.csv','../SVMscores/test_scores_u3.csv','Receiver Operating Characteristic curve')
    #plt.show()