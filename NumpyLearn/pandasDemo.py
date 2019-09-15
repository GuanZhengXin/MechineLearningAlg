import  pandas as pd
import numpy as np
from pandas import Series,DataFrame


df = pd.read_clipboard()

df_new = DataFrame(df,columns=['订单编号',"Test"],index=['A','B','C','D','E'])
df_new['订单编号'] = pd.Series([100,200],index=['A','B'])
df_new['Test'] = pd.Series(np.arange(10,15),index=['A','B','C','D','E'])
print(df_new)
print(df.columns)




