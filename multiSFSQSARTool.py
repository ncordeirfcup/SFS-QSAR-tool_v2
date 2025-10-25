import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
import os
from tkinter.filedialog import askopenfilename
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from kmca import kmca
from sequential_selection import stepwise_selection as sq
from loo import loo
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from rm2 import rm2
from applicability import apdom
import math
import numpy as np
#from WilliamsPlot import williams_plot
from matplotlib import pyplot
from stepwise_selection import stepwise_selection as ss
from sequential_selection import stepwise_selection as sq
from sequential_selection_lda import stepwise_selection as sqlda
from testset_prediction import testset_prediction as tsp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import threading


form = tk.Tk()
form.title("SFS-QSAR")
form.geometry("650x380")

tab_parent = ttk.Notebook(form)

tab1 = ttk.Frame(tab_parent)


tab_parent.add(tab1, text="Data preparation")


initialdir=os.getcwd()


reg=LinearRegression()

def data():
    filename = askopenfilename(initialdir=initialdir,title = "Select file")
    firstEntryTabOne.delete(0, END)
    firstEntryTabOne.insert(0, filename)
    global a_
    a_,b_=os.path.splitext(filename)
    global sfile
    sfile = pd.read_csv(filename)


def datav():
    filename1 = askopenfilename(initialdir=initialdir,title = "Select file with values to change")
    varEntryTabOne.delete(0, END)
    varEntryTabOne.insert(0, filename1)
    global c_
    c_,d_=os.path.splitext(filename1)
    global file1
    file1 = pd.read_csv(filename1, header=None)
    
def ptr():
    global file_pt
    filename_pt = askopenfilename(initialdir=initialdir,title = "Select pre-treated file")
    #NC2E.delete(0, END)
    #NC2E.insert(0, filename1)
    file_pt = pd.read_csv(filename_pt)
    return file_pt
        

import warnings
warnings.filterwarnings("ignore")



def process(tr,trX,Xts,y,yts,seed,num_des):
    lt=[]
    ls=['r2','neg_mean_absolute_error','neg_mean_poisson_deviance','neg_mean_gamma_deviance']
    #ls=['r2']
    lf=[0,5]
    #lf=[0]
    l1,l2,l3,l4,l5,l6,l7=[],[],[],[],[],[],[]
    xl=[]
    for i in ls:
        for j in lf:
            sqs=sq(trX,y,num_des,True,True,i,j)
            try:
               a1,b1=sqs.fit_()
               reg.fit(tr[a1],y)
               r2tr=reg.score(tr[a1],y)
               cv=loo(tr[a1],y,tr)
               xl.append(a1)
               print(str(len(xl))+' Model generated')
               c,m,l=cv.cal()
               r2pr,r2pr2,RMSEP=testpred(Xts[a1],yts,reg,m)
               rm2tr,drm2tr=rm2(y,l).fit()
               ytspr=pd.DataFrame(reg.predict(Xts[a1]))
               rm2ts,drm2ts=rm2(yts,ytspr).fit()
            except ValueError:
                c=0
                r2pr=0
                rm2tr=0
                rm2ts=0         
            #print(c,m,l)
            #print(a1)
            l1.append(i) 
            l2.append(j)  
            l3.append(c)
            l4.append(r2pr)
            l6.append(rm2tr)
            l7.append(rm2ts)
            l5.append(seed)
    Dict=dict([('random_seed', l5),('score', l1),('fold', l2),('Q2LOO', l3), ('R2Pred', l4), ('rm2tr', l6), ('rm2ts', l7)])
    #print(Dict)
    table=pd.DataFrame(Dict)
    return table

def correlation(X,cthreshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = X.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) > float(cthreshold)) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in X.columns:
                   del X[colname] # deleting the column from the dataset
    return X   

def variance(X,threshold):
    sel = VarianceThreshold(threshold=(threshold* (1 - threshold)))
    sel_var=sel.fit_transform(X)
    X=X[X.columns[sel.get_support(indices=True)]]    
    return X


def pretreat(X,cthreshold,vthreshold):
    X=correlation(X,cthreshold)
    X=variance(X,vthreshold)
    return X

def pretreated(df,cl,vr):
    X=df.iloc[:,2:]
    Xpt=pretreat(X,cl,vr)
    Xpt2=Xpt.columns.tolist()
    print('pre-treatment done')
    return df[Xpt2]

def testpred(Xts,yts,model,trav):
    ytspr=pd.DataFrame(model.predict(Xts))
    ytspr.columns=['Pred']
    tsdf=pd.concat([yts,pd.DataFrame(ytspr)],axis=1)
    tsdf.columns=['Active','Predict']
    tsdf['Aver']=trav
    tsdf['Aver2']=tsdf['Predict'].mean()
    tsdf['diff']=tsdf['Active']-tsdf['Predict']
    tsdf['diff2']=tsdf['Active']-tsdf['Aver']
    tsdf['diff3']=tsdf['Active']-tsdf['Aver2']
    r2pr=1-((tsdf['diff']**2).sum()/(tsdf['diff2']**2).sum())
    r2pr2=1-((tsdf['diff']**2).sum()/(tsdf['diff3']**2).sum())
    RMSEP=((tsdf['diff']**2).sum()/tsdf.shape[0])**0.5
    return r2pr,r2pr2,RMSEP

def random_div(df,Xpt2,per,ml,nd):
    nls=[]
    print(ml)
    for rs in ml:
        print('Random seed: '+str(rs)+': ')
        a,b= train_test_split(df,test_size=per, random_state=rs)
        tr=pd.DataFrame(a)
        ts=pd.DataFrame(b)
        tr=tr.reset_index().drop('index', axis=1)
        ts=ts.reset_index().drop('index', axis=1)
        trX=tr[Xpt2.columns]
        Xts=ts[Xpt2.columns]
        #print(trX.shape)
        y=tr.iloc[:,1:2]
        tr1=pd.concat([tr.iloc[:,0:2],trX],axis=1)
        ts1=pd.concat([ts.iloc[:,0:2],Xts],axis=1)
        yts=ts.iloc[:,1:2] 
        table=process(tr,trX,Xts,y,yts,rs,nd)
        nls.append(table)
        fd=pd.concat(nls, axis=0)
    return fd


def act_sort(df,Xpt2,per,ml,nd):
    nls=[]
    #ml=[i[0] for i in file1.values.tolist()]
    print(ml)
    for sp in ml:
        #perc=int(secondEntryTabOne.get())
        print('Activity sorting: '+str(sp))
        pc=int(100/per)
        filem=df.sort_values(df.iloc[:,1:2].columns[0],ascending=False)
        ts=filem.iloc[(sp-1)::pc, :]
        tr=filem.drop(ts.index.values)
        tr=tr.reset_index().drop('index', axis=1)
        ts=ts.reset_index().drop('index', axis=1)
        trX=tr[Xpt2.columns]
        Xts=ts[Xpt2.columns]
        y=tr.iloc[:,1:2]
        tr1=pd.concat([tr.iloc[:,0:2],trX],axis=1)
        ts1=pd.concat([ts.iloc[:,0:2],Xts],axis=1)
        print(tr1.shape)
        print(ts1.shape)
        yts=ts.iloc[:,1:2] 
        table=process(tr,trX,Xts,y,yts,sp,nd)
        nls.append(table)
        fd=pd.concat(nls, axis=0)
    return fd

def kmeansclust(df,Xpt2,per,ml,nd,nclus):
    nls=[]
    for rs in ml:
        print('Random seed: '+str(rs)+': ')
        kmc=kmca(df,per,int(rs),int(nclus))
        tr,ts=kmc.cal()
        tr=tr.reset_index().drop('index', axis=1)
        ts=ts.reset_index().drop('index', axis=1)
        trX=tr[Xpt2.columns]
        Xts=ts[Xpt2.columns]
        y=tr.iloc[:,1:2]
        tr1=pd.concat([tr.iloc[:,0:2],trX],axis=1)
        ts1=pd.concat([ts.iloc[:,0:2],Xts],axis=1)
        yts=ts.iloc[:,1:2] 
        table=process(tr,trX,Xts,y,yts,rs,nd)
        nls.append(table)
        fd=pd.concat(nls, axis=0)
    return fd
   
def datadiv():
    if NC1.get()=='no':
       cl=float(thirdEntryTabThreer3c2.get())
       vr = float(fourthEntryTabThreer5c2.get())
       Xpt2=pretreated(sfile,cl,vr)
       Xpt2.to_csv(str(a_)+'pretreated_file_'+str(cl)+'_'+str(vr)+'.csv', index=False)
    elif NC1.get()=='yes':
         file_pt=ptr()
         Xpt2=file_pt.iloc[:,1:]
         print(Xpt2.columns)
    per=int(secondEntryTabOne.get())
    ml=[i[0] for i in file1.values.tolist()]
    num_des=int(fifthBoxTabThreer6c2.get())
    if Selection.get()=='Activity sorting':      
       fd=act_sort(sfile,Xpt2,per,ml,num_des)
    elif Selection.get()=='Random':
       per=float(per)/100
       fd=random_div(sfile,Xpt2,per,ml,num_des)
    elif Selection.get()=='KMCA':
       nclus=int(cn.get())
       fd=kmeansclust(sfile,Xpt2,per,ml,num_des,nclus)
    fd['Average']=fd[['Q2LOO', 'R2Pred']].mean(axis=1)
    fd.sort_values('Average', ascending=False)
    if Selection.get()=='Activity sorting':
       fd.to_csv('Activity_sorting_results.csv', index=False)
    elif Selection.get()=='Random':
       fd.to_csv('Random_division_results.csv', index=False) 
    elif Selection.get()=='KMCA':
       fd.to_csv('KMCA_division_results.csv', index=False) 


def disable_clvar():
    thirdEntryTabThreer3c2['state']='disabled'
    fourthEntryTabThreer5c2['state']='disabled'

def enable_clvar():
    thirdEntryTabThreer3c2['state']='normal'
    fourthEntryTabThreer5c2['state']='normal'

def disable_cnvar():
    cn['state']='disabled'

def enable_cnvar():
    cn['state']='normal'
    


firstLabelTabOne = tk.Label(tab1, text="Select Data-matrix (.csv file)", font=("Helvetica", 12))
firstLabelTabOne.place(x=30,y=25)
firstEntryTabOne = tk.Entry(tab1,text='',width=50)
firstEntryTabOne.place(x=240,y=28)
b5=tk.Button(tab1,text='Browse', command=data, font=("Helvetica", 10))
b5.place(x=550,y=25)

varLabel = tk.Label(tab1, text="Variables for data-division", font=("Helvetica", 12))
varLabel.place(x=30,y=60)
newone = tk.Label(tab1, text="file (.csv file)", font=("Helvetica", 12))
newone.place(x=70,y=83)
varEntryTabOne = tk.Entry(tab1,text='',width=50)
varEntryTabOne.place(x=240,y=70)
b5=tk.Button(tab1,text='Browse', command=datav, font=("Helvetica", 10))
b5.place(x=550,y=68)


secondLabelTabOne_1=Label(tab1, text='Dataset division techniques', font=('Helvetica 12 bold'))
secondLabelTabOne_1.place(x=220,y=105)

secondLabelTabOne=Label(tab1, text='%Data-points(test set)',font=("Helvetica", 12), justify='center')
secondLabelTabOne.place(x=125,y=125)
secondEntryTabOne=Entry(tab1)
secondEntryTabOne.place(x=305,y=130)


Selection = StringVar()
Criterion_sel1 = ttk.Radiobutton(tab1, text='Activity sorting', variable=Selection, value='Activity sorting',command=disable_cnvar)
Criterion_sel2 = ttk.Radiobutton(tab1, text='Random Division', variable=Selection, value='Random',command=disable_cnvar)
Criterion_sel3 = ttk.Radiobutton(tab1, text='KMCA', variable=Selection, value='KMCA',command=enable_cnvar)
Criterion_sel1.place(x=50,y=165)
Criterion_sel2.place(x=200,y=165)
Criterion_sel3.place(x=350,y=165)
cn = Label(tab1, text= 'Cluster number',font=("Helvetica", 10))
cn.place(x=450,y=165)
cn= Spinbox(tab1, from_=0, to=100, width=5,state=DISABLED)
cn.place(x=550,y=165)


secondLabelTabOne_2=Label(tab1, text='Data pre-treatment/model development options', font=('Helvetica 12 bold'))
secondLabelTabOne_2.place(x=200,y=185)

NL1 = tk.Label(tab1, text='Do you have pretreated file: ',font=("Helvetica", 12),anchor=W, justify=LEFT)
NC1 = StringVar() 
NC1.set('no')
NC1_y = tk.Radiobutton(tab1, text='Yes', variable=NC1, value='yes', command=disable_clvar)
NC1_n = tk.Radiobutton(tab1, text='No', variable=NC1, value='no', command=enable_clvar)
NL1.place(x=100,y=215)
NC1_y.place(x=300,y=215)
NC1_n.place(x=370,y=215)


fifthLabelTabThreer6c2 = Label(tab1, text= 'Number of descriptors',font=("Helvetica", 12))
fifthLabelTabThreer6c2.place(x=30,y=250)
fifthBoxTabThreer6c2= Spinbox(tab1, from_=0, to=100, width=5)
fifthBoxTabThreer6c2.place(x=200,y=250)

thirdLabelTabThreer2c2=Label(tab1, text='Correlation cutoff', font=("Helvetica", 12))
thirdLabelTabThreer2c2.place(x=300,y=250)
thirdEntryTabThreer3c2=Entry(tab1)
thirdEntryTabThreer3c2.place(x=430,y=250)

fourthLabelTabThreer4c2=Label(tab1, text='Variance cutoff', font=("Helvetica", 12))
fourthLabelTabThreer4c2.place(x=250,y=280)
fourthEntryTabThreer5c2=Entry(tab1)
fourthEntryTabThreer5c2.place(x=370,y=280)


b6=tk.Button(tab1, text='Generate models', bg="orange", command=datadiv, font=("Helvetica", 10))
b6.place(x=280,y=310)

tab_parent.pack(expand=1, fill='both')

form.mainloop()


