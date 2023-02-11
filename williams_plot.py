import pandas as pd
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image
#import pymysql
import os
import shutil
import numpy as np
from tkinter.filedialog import askopenfilename
from numpy.linalg import matrix_power
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt


initialdir=os.getcwd()

def data1():
    global filename1
    filename1 = askopenfilename(initialdir=initialdir,title = "Training set file")
    firstEntryTabThree.delete(0, END)
    firstEntryTabThree.insert(0, filename1)
    global c_
    c_,d_=os.path.splitext(filename1)
    global file1
    file1 = pd.read_csv(filename1)
    global col1
    col1 = list(file1.head(0))

def data2():
    global filename2
    filename2 = askopenfilename(initialdir=initialdir,title = "Select test set file")
    secondEntryTabThree.delete(0, END)
    secondEntryTabThree.insert(0, filename2)
    global file2
    file2 = pd.read_csv(filename2)



def process():
    #Scripts of the Willams plot were conceptualized from the Jupyter file provided with the article https://doi.org/10.1016/j.scitotenv.2022.160590
    nd=int(thirdEntryTabThreer3c1.get())
    print(nd)
    dftr=file1.iloc[:,:(nd+3)]
    try:
       dfts=file2[dftr.columns]
    except:
       print('Error: Mismatch in the descriptors of training and test sets')
    Y_train=dftr.iloc[:,-2:-1]
    Y_test=dfts.iloc[:,-2:-1]
    X_train=dftr.iloc[:,1:-2]
    X_test=dfts.iloc[:,1:-2]
    Yt_pred=dftr.iloc[:,-1:]
    Yv_pred=dfts.iloc[:,-1:]
    dftr['Residual']=dftr[Y_train.columns[0]]-dftr[Yt_pred.columns[0]]
    dfts['Residual']=dfts[Y_test.columns[0]]-dfts[Yv_pred.columns[0]]
    scaler_res = StandardScaler()
    st_residuals_train = scaler_res.fit_transform(dftr.iloc[:,-1:])
    st_residuals_val = scaler_res.transform(dfts.iloc[:,-1:])
    res_tr=pd.DataFrame(st_residuals_train, columns=['Standardized Residuals'])
    res_ts=pd.DataFrame(st_residuals_val, columns=['Standardized Residuals'])
    h_k = (3 * int(len(X_train.columns)+1))/len(X_train.index)
    print(h_k)

    inner = matrix_power((np.dot(X_train.T,X_train)), -1)
    h_indexes = []
    for idx in X_train.index.values:
        xi_t = np.asarray(X_train.loc[idx]).T
        xi = np.asarray(X_train.loc[idx])
        hi = np.dot(xi_t.dot(inner), xi)
        h_indexes.append(hi)
    
    h_indexes_v = []
    for idx in X_test.index.values:
        xi_t = np.asarray(X_test.loc[idx]).T
        xi = np.asarray(X_test.loc[idx])
        hi_v = np.dot(xi_t.dot(inner), xi)
        h_indexes_v.append(hi_v)

    htr=pd.DataFrame(h_indexes, columns=['Leverage'])    
    hts=pd.DataFrame(h_indexes_v, columns=['Leverage'])   

    ftr=pd.concat([dftr,res_tr,htr], axis=1)
    fts=pd.concat([dfts,res_ts,hts], axis=1)

    ftr.to_csv('training_out.csv', index=False)
    fts.to_csv('test_out.csv', index=False)

    plt.figure(figsize=(15,10))
    plt.scatter(h_indexes, st_residuals_train, s=120, marker='s', edgecolors='k', c="tab:blue",alpha=0.6)
    plt.scatter(h_indexes_v, st_residuals_val, s=120, marker='o', edgecolors='k',c="tab:red", alpha=0.6)

    plt.ylim(-4, 4)
    plt.xlim(right=1)
    plt.xlim(left=0)

    leg = list()
    leg.append('Train')
    leg.append('Test')
    
    plt.legend(leg, loc='best', frameon=True, fontsize=16)
    plt.xlabel('Leverages', fontsize=20)
    plt.ylabel('Standarized residuals', fontsize=20)
    plt.axhline(y=3, color='gray', linestyle='--')
    plt.axhline(y=-3, color='gray', linestyle='--')
    plt.axvline(x=h_k, linestyle='--')
    plt.tight_layout()
    #plt.show()
    plt.savefig('williams_ad.jpg', dpi=300)


form = tk.Tk()

form.title("SFS-QSAR-WiiliamsPlot")

form.geometry("690x180")


tab_parent = ttk.Notebook(form)


tab1 = tk.Frame(tab_parent) #background='#ffffff')


tab_parent.add(tab1, text="SFS-QSAR-Williams_plot")


firstLabelTabThree = tk.Label(tab1, text="Select training set",font=("Helvetica", 12))
firstLabelTabThree.place(x=90,y=10)
firstEntryTabThree = tk.Entry(tab1, width=40)
firstEntryTabThree.place(x=230,y=13)
b3=tk.Button(tab1,text='Browse', command=data1,font=("Helvetica", 10))
b3.place(x=480,y=10)

secondLabelTabThree = tk.Label(tab1, text="Select test set",font=("Helvetica", 12))
secondLabelTabThree.place(x=120,y=40)
secondEntryTabThree = tk.Entry(tab1,width=40)
secondEntryTabThree.place(x=230,y=43)
b4=tk.Button(tab1,text='Browse', command=data2,font=("Helvetica", 10))
b4.place(x=480,y=40)

thirdLabelTabThreer2c1=Label(tab1, text='Number of descriptors',font=("Helvetica", 12))
thirdLabelTabThreer2c1.place(x=120,y=70)
thirdEntryTabThreer3c1=Entry(tab1)
thirdEntryTabThreer3c1.place(x=300,y=70)


b4=Button(tab1, text='Submit', command=process,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b4.place(x=330,y=100)

NL1=Label(tab1, text='Please note: The input training and test set files should be collected after generating the model with SFS-QSAR from respective folders',font=("Helvetica", 8))
NL1.place(x=10,y=130)



tab_parent.pack(expand=1, fill='both')
form.mainloop()
