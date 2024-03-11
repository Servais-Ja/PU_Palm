#1.8*10^-4
import numpy as np

fscore=np.loadtxt('feature_importances/feature_importances.txt')
posfall=np.loadtxt('Feature/allFeature/allpos.txt')
unfall=np.loadtxt('Feature/allFeature/allun.txt')
posfinal1=np.loadtxt('Feature/seqFeature/x_cksaagp_gaac_p.txt')
unfinal1=np.loadtxt('Feature/seqFeature/x_cksaagp_gaac_u.txt')

posfall=posfall.T
unfall=unfall.T

#idx=np.where(fscore>=2.2*10**(-4))#200
#idx=np.where(fscore>=2.37*10**(-4))#150
#idx=np.where(fscore>=2.72*10**(-4))#100
#idx=np.where(fscore>=3.47*10**(-4))#50
idx=np.where(fscore[:3040]>=3.39*10**(-4))#pssm30
posfinal=posfall[idx]
unfinal=unfall[idx]
"""
np.savetxt('Feature/allFeature/chosenpssm30pos.txt',posfinal.T)
np.savetxt('Feature/allFeature/chosenpssm30un.txt',unfinal.T)
"""

posfinal=np.hstack((posfinal.T,posfinal1))
unfinal=np.hstack((unfinal.T,unfinal1))
np.savetxt('Feature/allFeature/finalpos.txt',posfinal)
np.savetxt('Feature/allFeature/finalun.txt',unfinal)
