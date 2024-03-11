import model_pr,model_roc
poslist=[]
unlist=[]
namelist=['final']
for i in namelist:
    poslist.append('../SVMscores/scores_p'+i+'.csv')
    unlist.append('../SVMscores/scores_u'+i+'.csv')

for i in range(len(namelist)):
    model_pr.main_pr(poslist[i],unlist[i],namelist[i])
    model_roc.main_roc(poslist[i],unlist[i],namelist[i])

#plt.show()