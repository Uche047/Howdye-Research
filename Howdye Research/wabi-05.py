#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
from pathlib import Path
from scipy.interpolate import interp1d
import lasio as ls
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


import Loading_Well


# In[3]:


well = ls.read(r'wabi-05.las')
df = well.df() 


# In[4]:


df = df.reset_index()


# In[5]:


df


# In[6]:


df.head(10)


# In[7]:


df.columns.values


# Dropping unwanted logs

# In[20]:


df_New = df [['MD', 'AHT10_3', 'AHT20_1', 'AHT30_1', 'AHT60_1', 'AHT90_1',
       'AHT90_3', 'APLC_1', 'APLC_2', 'APLC_4', 'BS_2', 'COGR_1',
       'COGR_2', 'DTCO_1', 'DTCO_2', 'DTP_2', 'DTS_2', 'DTSM_1', 'DTSM_2',
       'ECGR_2', 'FPLC_1', 'GR_1', 'GR_OBMI_1', 'HCAL_1', 'HCAL_2',
       'HCAL_4', 'HCGR_1', 'HDRA_1', 'HDRA_2', 'HG-IP_1', 'HG-IS_1',
       'HG-POISSON_1', 'HG-ZOEP-REFL-EI-0_1', 'HG-ZOEP-REFL-STACK-0-20_1',
       #'HG-ZOEP-REFL-STACK-20-40_1', 'HSGR_1', 'HSGR_2',
       #'JPD_26092007IP_1', 'JPD_26092007VP_1',
       #'JPD_26092007ZOEP-REFL-EI-0_1', 'LCD_AHT10_1', 'LCD_AHT90_1',
       'LCD_APLC_1', 'LCD_BS_1', 'LCD_DTP_1', 'LCD_DTS_1', 'LCD_ECGR_1',
       #'LCD_GR_1', 'LCD_HCAL_1', 'LCD_PEFZ_1', 'LCD_PHIE_1', 'LCD_PHIT_1',
       'LCD_RHOZ_1', 'LCD_SWE_1', 'LCD_SWT_1', 'LCD_SXOE_1', 'LCD_VCL_1',
       #'LS-IP_1', 'LS-VP_1', 'LS-ZOEP-REFL-EI-0_1', 'MD_1',
       #'P_S_LCD_JPD_24102007IP_1', 'P_S_LCD_JPD_24102007POISSON_1',
       #'P_S_LCD_JPD_24102007VP_1', 'P_S_LCD_JPD_24102007VPONVS_1',
       #'P_S_LCD_JPD_24102007VS_1', 'P_S_LCD_JPD_24102007ZOEP-REFL-EI-0_1',
       'PEFZ_1', 'PEFZ_2', 'PEFZ_4', 'PHIE_1', 'PHIT_1', 'POISSON_1',
       #'PRES_1', 'RHOZ_1', 'RHOZ_2', 'RHOZ_4', 'RHOZ-V4-BLOCK-10_1',
       #'SBD2_1', 'SBD2_2', 'SCO2_1', 'SCO2_2', 'SDDE_1', 'SDDE_2',
       #'SDDE_3', 'SEDA_1', 'SEDA_2', 'SEDP_1', 'SEDP_2', 'SEDP_3',
       #'SEMA_1', 'SEMA_2', 'SEMP_1', 'SEMP_2', 'SESA_1', 'SESA_2',
       #'SESP_1', 'SESP_2', 'SEXA_1', 'SEXA_2', 'SEXP_1', 'SEXP_2',
       'SGRC_1', 'SGRC_2', 'SGRC_3', 'SKAL_1', 'SKAL_2', 'SWE_1', 'SWT_1',
       #'SYNTH-FROM-TEST_19092007ZOEP-REFL-EI-0_1', 'TENS_1', 'TENS_2',
       #'TEST2MS_POISSON_11102007POISSON_1',
       #'TEST2MS_POISSON_11102007ZOEP-REFL-EI-30_1',
       #'TEST3_ZOEP10DEGRE_11102007ZOEP-REFL-EI-10_1', 'TEST_19092007IP_1',
       #'TEST_19092007VP_1', 'TEST_19092007ZOEP-REFL-EI-0_1',
       #'TESTPOISSON11102007POISSON_1',
       #'TESTPOISSON11102007ZOEP-REFL-EI-30_1', 'TFP_LOG_1',
       'TFP_VELOCITY_LOG_1', 'TNPL_1', 'TNPL_2', 'TNPL_3', 'TVD_1',
       'TVDSS_1', 'VCL_1', 'VPVS_1', 'WAV-RICKER_25HZ_ROT0_1_1',
       'WAV-RICKER_25HZ_ROT0_1_2', 'ZOEP-REFL-EI-0_1']].copy()


# In[9]:


df 


# In[10]:


df.keys()


# In[11]:


df1= df[['MD','GR_1','LCD_SWE_1','LCD_SWT_1','DTCO_1','SWT_1','TVDSS_1','PHIE_1', 
         'PHIT_1','AHT20_1', 'AHT30_1', 'AHT60_1', 'AHT90_1']]


# In[12]:


df1


# # Data Cleaning

# In[13]:


df1.describe()


# In[14]:


df1.isnull().sum()


# In[17]:


fig,axs=plt.subplots(1,2, figsize=(5,12), sharey=True)

ax=axs[0]
ax.scatter(df1['GR_1'], df1['MD'],s=1)
ax.set_xlim(0,130)
ax.set_xlabel('GR [gAPI]', fontsize='large')
ax.set_ylabel('Measured Depth [mbsf]', fontsize='large')
ax.grid()
ax.fill_betweenx(df1['MD'],df1['GR_1'],70,
                           where=df1['GR_1']>70,interpolate=True,color='gray', label='shales')
ax.fill_betweenx(df1['MD'],df1['GR_1'],70,
                           where=df1['GR_1']<70,interpolate=True,color='y', label='sands')
ax.axvline(70,color='k',linewidth=1,linestyle='--')
ax.spines["top"].set_position(("axes", 1.0))
ax.legend()
ax.invert_yaxis()

ax=axs[1]
ax.scatter(df1['DTCO_1'], df1['MD'],s=1)
ax.set_xlim(0,130)
ax.set_xlabel('DT [ft/s]', fontsize='large')
ax.grid()
ax.spines["top"].set_position(("axes", 1.0))
ax.legend()


# In[19]:


fig,axs=plt.subplots(1,2, figsize=(5,8), sharey=True)

ax=axs[0]
ax.scatter(df1['SWT_1'], df1['MD'],s=1)
ax.set_xlim(0,1.0)
ax.set_xlabel('m3/m3', fontsize='large')
ax.grid()
#ax.fill_betweenx(df1['MD'],df1['SWT_1'],#0.25,
#                           where=df1['SWT_1']>0.25,interpolate=True,color='white', label='water')
#ax.fill_betweenx(df1['MD'],df1['SWT_1'],0.25,
 #                          where=df1['SWT_1']<0.25,interpolate=True,color='y', label='oil')
ax.axvline(0.25,color='k',linewidth=1,linestyle='--')
ax.spines["top"].set_position(("axes", 1.0))
ax.legend()
ax.invert_yaxis()


# ## Data Cleaning

# In[22]:


df_New.isnull().sum()


# In[ ]:





# In[24]:


df_without_NaN = df_New.dropna() # drop null values
df_without_NaN.isnull().sum()


# In[56]:


#bfill, ffill,
arr = [2, 3, np.nan, 4,5, 67, 56,8,23, np.nan, 45, np.nan, 8]
arr1 = np.array(arr)
#arr1 = arr.bfill(inplace = True) # bfill and ffill are functions for pandas
arr1 = pd.DataFrame(arr1)
#print(arr1.mean())
##arr1 = arr1.fillna(method = 'ffill')
arr2 = arr1.fillna(arr1.median(), inplace=False)
arr2
#arr1.mean()
# mean , std, median, mode


# ARCHIE'S EQUATION 
# $$
# S_w  = \Biggr(\frac{a . R_w}{\phi^m . R_t}\Biggl)^{(\frac{1}{n})}
# $$
# 
# R_w = Resistivity of water      
# R_t  = True formation resistivity     
# phi = is the rocks porosity   
# m cementation factor normally  a value of 2 is assumed   
# a = cementation exponent (1.8 - 2)    
# n = Saturation exponent (2)    

# In[57]:


df_without_NaN.columns.values


# In[59]:



def Water_Saturation(Porosity, Res_Lith, Res_Fluid, a, m, n):
    """
    docstring
    Poosity: Porosity
    """
    
    numerator  = a *  Res_Fluid
    denominator = Porosity**m * Res_Lith
    
    return (numerator/denominator)**(1/n)
    


# In[66]:


df_without_NaN['AHT90_3'].describe()


# In[ ]:


cutoff = 65
for i in range(len(df_without_NaN)):
    if gamma_ray < cutoff & Vclay> 0.5 :
        def Water_Saturation_Archie(Porosity, Res_Lith, Res_Fluid, a, m, n):
        """
        docstring
        Poosity: Porosity
        """

        numerator  = a *  Res_Fluid
        denominator = Porosity**m * Res_Lith

        return (numerator/denominator)**(1/n)
    else:
        
        def Water_Saturation_Simansouda(Porosity, Res_Lith, Res_Fluid, a, m, n):
        """
        docstring
        Poosity: Porosity
        """

        numerator  = a *  Res_Fluid
        denominator = Porosity**m * Res_Lith

        return (numerator/denominator)**(1/n)
        
    


# In[64]:


df_without_NaN['Water_Saturation'] = Water_Saturation(Porosity=df_without_NaN['PHIT_1'],
                                                     Res_Lith=df_without_NaN['AHT90_3'],
                                                     Res_Fluid=20,
                                                     a = 2, m = 2, n= 2)
df_without_NaN['Water_Saturation']


# In[63]:


fig, ax =plt.subplots()
ax.plot(df_without_NaN['Water_Saturation'], df_without_NaN['MD'])
ax.invert_yaxis()


# In[ ]:




