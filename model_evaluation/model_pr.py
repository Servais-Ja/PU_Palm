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

def main_pr(pospath,unpath,title):
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

    # we can directly use the contingency tables we already computed anyways
    pr_bounds_function = lambda tables: ss._lb_ub(lower=ss.pr_from_cts(tables.lower),
                                      upper=ss.pr_from_cts(tables.upper))
    pr_bounds = pr_bounds_function(tables)

    _, pr_neg = optunity.metrics.pr_auc(labels, decision_values, return_curve=True)


    xs = [float(x) / 100 for x in range(101)]
    pr_up = ss.zoh(*zip(*pr_bounds.upper))
    pr_lo = ss.zoh(*zip(*pr_bounds.lower))
    fig = plt.figure(3)
    fix_plot_shape(fig)
    plt.plot(*zip(*pr_neg), color='red')#, linestyle='dashed'
    plt.fill_between(xs, list(map(pr_lo, xs)), list(map(pr_up, xs)), color='orange', alpha=0.4)
    #plot_proxy()
    #plot_proxy2()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.legend(['beta=0','expected region via betahat'],loc="upper left", bbox_to_anchor=(1,1))#图例
    plt.title(title)
    plt.show()

if __name__=='__main__':
    main_pr('../SVMscores/scores_pfinal.csv','../SVMscores/scores_ufinal.csv','Precision-Recall curve')
    #plt.show()