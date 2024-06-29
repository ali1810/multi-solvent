import streamlit as st
from PIL import Image
from stmol import showmol
import py3Dmol
import re
import numpy as np 
from rdkit.Chem import AllChem
import pubchempy as pcp
import joblib
import streamlit as st
import pickle
from PIL import Image
import pandas as pd
from rdkit import Chem
#from rdkit.Chem import Draw 
import xgboost
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import base64
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import MolFromSmiles, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
import streamlit as st
import base64
from streamlit_shap import st_shap
import shap
from xgboost import XGBRegressor
import xgboost as xgb
from urllib.request import urlopen
from PIL import Image
#from flask.wrappers import Request
#import threading
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.header("------Multi-Solvent Solubility Prediction------")
        
def calculate_aromatic_proportion(smiles):
    # Parse SMILES string and generate molecular representation
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Invalid SMILES
    
    # Identify aromatic atoms
    aromatic_atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms() if atom.GetIsAromatic()]
    
    # Calculate aromatic proportion
    total_atoms = mol.GetNumAtoms()
    aromatic_proportion = len(aromatic_atoms) / total_atoms
    
    return aromatic_proportion


SMILES = st.text_input('**Type SMILES below**👇...then press predict button',value = "CC(=O)OC1=CC=CC=C1C(=O)O")
aromatic_proportion = calculate_aromatic_proportion(SMILES)
if aromatic_proportion is not None:
    st.write("Given smiles is Valid")
else:
    st.write("Invalid SMILES")

solvents = {  
         'n-octanol':  'CCCCCCCCO' ,  
              'DMSO':  'CS(C)=O'    , 
   'ethylene glycol':    'OCCO',
         'n-hexane':     'CCCCCC',
        'chloroform':    'ClC(Cl)Cl',
      'sec-butanol':    'CCC(C)O',
               'NMP':     'CN1CCCC1=O',
    'methyl acetate':     'COC(C)=O',
       'n-pentanol':     'CCCCCO',
       'cyclohexane':    'C1CCCCC1',
               'THF':     'C1CCOC1' , 
        '2-butanone':     'CCC(C)=O',
        'isobutanol':    'CC(C)CO' ,
        '1,4-dioxane':    'C1COCCO1',
               'DMF':    'CN(C)C=O',
          'toluene':    'Cc1ccccc1',
     'acetonitrile':    'CC#N',
         'n-butanol':    'CCCCO',
            'water':    'O',
       'n-propanol':    'CCCO',
         'acetone'  :  'CC(C)=O',
     'ethyl acetate':    'CCOC(C)=O',
       'isopropanol':    'CC(C)O',
          'methanol':    'CO',
           'ethanol':    'CCO'  
}
# Title of the web app
#st.title('Solvent SMILES Selector')

# Dropdown menu for selecting a solvent
selected_solvent = st.selectbox('Select a solvent:', list(solvents.keys()))
temperature = st.number_input('Enter the temperature (°C):', step=5.0)

# Get the SMILES notation for the selected solvent
solvent_smiles = solvents[selected_solvent]

# Display the selected solvent and its SMILES notation
#st.write(f'You selected: {selected_solvent}')
#st.write(f'SMILES notation: {solvent_smiles}')

prop=pcp.get_properties([ 'MolecularWeight'], SMILES, 'smiles')
x = list(map(lambda x: x["CID"], prop)) 
y=x[0]
    #print(y) 
x = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/PNG?image_size=300x300"
url=(x % y)
#print(url)
img = Image.open(urlopen(url))
mol = Chem.MolFromSmiles(SMILES)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
mblock = Chem.MolToMolBlock(mol)
xyzview = py3Dmol.view(width=300,height=300)
    #xyzview = py3Dmol.view(query=′pdb:1A2C′)
xyzview.addModel(mblock,'mol')
xyzview.setStyle({'model': -1}, {"cartoon": {'color': 'spectrum'}})
style = 'stick'
spin = st.checkbox('Animation', value = True) 
#style = st.selectbox('Chemical structure',['stick','ball-and-stick','line','cross','sphere']) 

#bcolor = st.sidebar.color_picker('Pick background Color', '#0C0C0B')
xyzview.spin(True)
if spin:
    xyzview.spin(True)
else:
    xyzview.spin(False)
    #xyzview.setStyle({'sphere':{}})
xyzview.setBackgroundColor('#EAE5E5')
xyzview.zoomTo()
xyzview.setStyle({style:{'color':'spectrum'}})
    


col1, mid, col2 = st.columns([15,2.5,15])
#col1, mid, col2 = st.columns([15,0.5,15])
with col1:
        st.image(img, use_column_width=False)
with col2:
        showmol(xyzview,height=300,width=400) 
if st.button("Predict"):
	        #prop=pcp.get_properties([ 'MolecularWeight'], SMILES, 'smiles')
              x = list(map(lambda x: x["CID"], prop))
              y=x[0]
              x = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/%s/xml"
              data=requests.get(x % y)
     
              html = BeautifulSoup(data.content, "xml")
              solubility = html.find(name='TOCHeading', string='Solubility')
              if solubility ==None:
                    sol= None

              else:
                solub=solubility.find_next_sibling('Information').find(name='String').string
                sol= solub
                #st.write("Work is in progress...cccc::", sol)
              mol = Chem.MolFromSmiles(SMILES)
    
               # calculate the log octanol/water partition descriptor
              single_MolLogP = Descriptors.MolLogP(mol)
    
              # calculate the molecular weight descriptor
              single_MolWt   = Descriptors.MolWt(mol)
    
             # calculate of the number of rotatable bonds descriptor
              single_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
              #m = Chem.MolFromSmiles(elem)
              aromatic_list = [mol.GetAtomWithIdx(i).GetIsAromatic() for i in range(mol.GetNumAtoms())]
              aromatic = 0
              for i in aromatic_list:
                 if i:
                   aromatic += 1
              heavy_atom = Lipinski.HeavyAtomCount(mol)
              single_AP=aromatic / heavy_atom if heavy_atom != 0 else 0




             # Calculate ring count 
              single_RC= Descriptors.RingCount(mol)

             # Calculate TPSA 
              single_TPSA=Descriptors.TPSA(mol)

             # Calculate H Donors  
              single_Hdonors=Lipinski.NumHDonors(mol)

            # Calculate saturated Rings 
              single_SR= Lipinski.NumSaturatedRings(mol) 

            # Calculate Aliphatic rings 
              single_AR =Lipinski.NumAliphaticRings(mol)
                  # Calculate Hydrogen Acceptors 
              single_HA = Lipinski.NumHAcceptors(mol)

              # Calculate Heteroatoms
              single_Heter = Lipinski.NumHeteroatoms(mol)

              
              single_Max_Partial_Charge =  Descriptors.MaxPartialCharge(mol)
              single_FP_density =  Descriptors.FpDensityMorgan1(mol)
              single_num_valence_electrons = Descriptors.NumValenceElectrons(mol)
              single_NHOH_count = Lipinski.NHOHCount(mol)
              single_SP3_frac = Lipinski.FractionCSP3(mol)
              single_SP_bonds = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[^1]'))) 
             # put the descriptors in a list
              rows = np.array([single_MolLogP, single_MolWt, single_NumRotatableBonds, single_AP,single_RC,
                       single_TPSA,single_Hdonors,single_SR,single_AR,
                      single_HA,single_Heter,single_Max_Partial_Charge,single_FP_density,single_num_valence_electrons,
                     single_NHOH_count,single_SP3_frac,single_SP_bonds])
    
             # add the list to a pandas dataframe
             #single_df = pd.DataFrame(single_list).T
              baseData = np.vstack([rows])
              # rename the header columns of the dataframe
    
                #columnNames = ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion","Ring_Count","TPSA","H_donors","Saturated_Rings","AliphaticRings","H_Acceptors","Heteroatoms"]
              columnNames = ["MolP","MolWt", 
                   "NumRotatableBonds", "AromaticProportion"
                  ,"Ring_Count","TPSA","H_donors", "Saturated_Rings","AliphaticRings","H_Acceptors","Heteroatoms","Max_Partial_Charge",
                  "valence_electrons","FP_density","NHOH_count","SP3_frac","SP_bonds"]
 
              descriptors = pd.DataFrame(data=baseData, columns=columnNames)
              #generated_descriptors1= predictSingle(smiles)
              mol1 = Chem.MolFromSmiles(solvent_smiles)
    
               # calculate the log octanol/water partition descriptor
              single_MolLogP = Descriptors.MolLogP(mol1)
    
              # calculate the molecular weight descriptor
              single_MolWt   = Descriptors.MolWt(mol1)
    
             # calculate of the number of rotatable bonds descriptor
              single_NumRotatableBonds = Descriptors.NumRotatableBonds(mol1)
              aromatic_list = [mol1.GetAtomWithIdx(i).GetIsAromatic() for i in range(mol1.GetNumAtoms())]
              aromatic = 0
              for i in aromatic_list:
                if i:
                 aromatic += 1
                 heavy_atom = Lipinski.HeavyAtomCount(mol)
              #return aromatic / heavy_atom if heavy_atom != 0 else 0
             # calculate the aromatic proportion descriptor
              try:
                single_AP = aromatic / heavy_atom if heavy_atom != 0 else 0
              except ZeroDivisionError:
                single_AP = 0 
                #single_AP = aromatic / heavy_atom if heavy_atom != 0 else 0

             # Calculate ring count 
              single_RC= Descriptors.RingCount(mol1)

             # Calculate TPSA 
              single_TPSA=Descriptors.TPSA(mol1)

             # Calculate H Donors  
              single_Hdonors=Lipinski.NumHDonors(mol1)

            # Calculate saturated Rings 
              single_SR= Lipinski.NumSaturatedRings(mol1) 

            # Calculate Aliphatic rings 
              single_AR =Lipinski.NumAliphaticRings(mol1)
                  # Calculate Hydrogen Acceptors 
              single_HA = Lipinski.NumHAcceptors(mol1)

              # Calculate Heteroatoms
              single_Heter = Lipinski.NumHeteroatoms(mol1)

              
              single_Max_Partial_Charge =  Descriptors.MaxPartialCharge(mol1)
              single_FP_density =  Descriptors.FpDensityMorgan1(mol1)
              single_num_valence_electrons = Descriptors.NumValenceElectrons(mol1)
              single_NHOH_count = Lipinski.NHOHCount(mol1)
              single_SP3_frac = Lipinski.FractionCSP3(mol1)
              single_SP_bonds = len(mol1.GetSubstructMatches(Chem.MolFromSmarts('[^1]')))
            
                        
    

              # put the descriptors in a list
              rows = np.array([single_MolLogP, single_MolWt, single_NumRotatableBonds, single_AP,single_RC,
                       single_TPSA,single_Hdonors,single_SR,single_AR,
                      single_HA,single_Heter,single_Max_Partial_Charge,single_FP_density,single_num_valence_electrons,
                     single_NHOH_count,single_SP3_frac,single_SP_bonds])
    
             # add the list to a pandas dataframe
             #single_df = pd.DataFrame(single_list).T
              baseData = np.vstack([rows])
              # rename the header columns of the dataframe
    
                #columnNames = ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion","Ring_Count","TPSA","H_donors","Saturated_Rings","AliphaticRings","H_Acceptors","Heteroatoms"]
              columnNames = ["MolP1","MolWt1", 
                   "NumRotatableBonds1", "AromaticProportion1"
                  ,"Ring_Count1","TPSA1","H_donors1", "Saturated_Rings1","AliphaticRings1","H_Acceptors1","Heteroatoms1","Max_Partial_Charge1",
                  "valence_electrons1","FP_density1","NHOH_count1","SP3_frac1","SP_bonds1"]
 
              descriptors1 = pd.DataFrame(data=baseData, columns=columnNames)
              input_data = pd.concat([descriptors, descriptors1],axis=1)
              input_data['temprature_celsius'] = temperature
              #input_df = pd.DataFrame([input_data])
              #return input_df
              loaded_bst = xgb.Booster()
              loaded_bst.load_model('xgboost_model1.bin')

# Predict on the test set using the loaded model
##preds = loaded_bst.predict(dtest)
              df3 = xgb.DMatrix(input_data)

              
              #loaded_model = joblib.load('xgbboost_model1.bin')
              Gram_liter1=loaded_bst.predict(df3)
              #Gram_liter1=prediction[0]
              MolWt1     =descriptors["MolWt"]
              #return prediction[0] 
              #
              
              
              
              
              
              
              #mol = Chem.MolFromSmiles(SMILES)
              #fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
              #arr = np.zeros((0,), dtype=np.int8)
              #arr1=DataStructs.ConvertToNumpyArray(fp,arr)
              #arr2 =pd.DataFrame(arr)
              #array = arr2.T
    #print(array.shape)
#print(fingerprints_array1)
#mols = [Chem.rdmolfiles.MolFromSmiles(SMILES_string) for SMILES_string in smiles]
              #loaded_model = joblib.load('/content/drive/MyDrive/xgb_model.pkl')
              #trained_model= xgb.Booster()
              #trained_model.load_model('model_xgb_95 2.bin')
              #trained_model.load_model('model_xgb_95 2.bin')
              #df3=pd.concat([array,descriptors1],axis=1)
              #df3 = xgb.DMatrix(df3)
	
#print(df3)
              #pred_rf1 = trained_model.predict(df3)
              #pred_rf1 =  (pred_rf1-0.18)
              #pred_rf2 =  np.round(pred_rf1,2)	
              # Convert g/L to mol/L
              mol_liter1 = Gram_liter1 / MolWt1
              mol_liter2   = np.round(mol_liter1,4)
# Calculate the log base 10 of the concentration in mol/L
              pred_rf1 = np.log10(mol_liter1)# if mol_liter1 > 0 else None


              
              #mol_liter1   =  10**pred_rf1
              

    	        #compounds = pcp.get_compounds(SMILES, namespace='smiles')
              #match = compounds[0]
              #c_name =match.iupac_name

              #MolWt1     =descriptors1["MolWt"]
              #Gram_liter1  =pred_rf1
              #Gram_liter1  =(10**pred_rf1)*MolWt1
              #Gram_liter1 = round(Gram_liter1,2) 	
               #P_sol1 =smiles_to_sol(smiles) ## calling function to get the solubility from <pubchem
#df_results = pd.DataFrame(df_results1)
    #render_mol(blk)
              data = dict(SMILES=SMILES, Predicted_LogS=pred_rf1, Mol_Liter=mol_liter2,Gram_Liter=Gram_liter1,Experiment_Solubility_PubChem=sol)
              df = pd.DataFrame(data, index=[0])
              #selected_solvent
              #st.write(f'Predicted Solubility for: {selected_solvent}',f'at : {temperature}')°C)
              #st.write(f'Predicted Solubility for: {selected_solvent}',and temperature is : {temperature} °C')
              #st.write(f'Entered temperature: {temperature} °C, Concentration: {concentration_mol_L:.6f} mol/L' if concentration_mol_L > 0 else 'Concentration: 0 mol/L')
              #st.write(f'Predicted Solubility in {selected_solvent},at temperature: {temperature}°C') 
              #st.write(f"Concentration: <span style='color:green; font-weight:bold;'>{concentration_mol_L:.6f} mol/L</span>", unsafe_allow_html=True)
              #st.write(f"Predicted Solubility in: <span style='color:green; font-weight:bold;'>{selected_solvent},at temperature: <span style='color:green; font-weight:bold;'>{temperature}°C", unsafe_allow_html=True)
              st.write(f"Predicted Solubility in: <span style='color:green; font-weight:bold;'>{selected_solvent} </span>, at temprature : <span style='color:green; font-weight:bold;'>{temperature} °C</span>", unsafe_allow_html=True)
              ####
              # CSS for dark themed table
              
              #st.write('Predicted Solubility for single smiles')
              st.table(df.style.format({"Predicted_LogS": "{:.2f}","Mol_Liter":"{:.2f}","Gram_Liter":"{:.2f}"}))
              st.write('Computed molecular descriptors')
              df1 = pd.DataFrame(descriptors, index=[0])
              #descriptors1 # Skips the dummy first item
              st.table(df1)
def getAromaticProportion(m):
         aromatic_list = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
         aromatic = 0
         for i in aromatic_list:
           if i:
             aromatic += 1
           heavy_atom = Lipinski.HeavyAtomCount(m)
         return aromatic / heavy_atom if heavy_atom != 0 else 0
st.write("""---------**OR**---------""")
st.write("""**Upload a 'csv' file with a column named 'SMILES'** (Max:2000)""")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    # data
        SMILES=data["SMILES"]  
      
if st.button('Prediction for input file'):


           moldata = []
           for elem in SMILES:
              mol = Chem.MolFromSmiles(elem)
              moldata.append(mol)

           baseData = np.arange(1, 1)
           i = 0
           for mol in moldata:
               desc_MolLogP = Descriptors.MolLogP(mol)
               desc_MolWt = Descriptors.MolWt(mol)
               desc_NumRotatableBonds = Lipinski.NumRotatableBonds(mol)
               desc_AromaticProportion = getAromaticProportion(mol)
               desc_Ringcount        =   Descriptors.RingCount(mol)
               desc_TPSA = Descriptors.TPSA(mol)
               desc_Hdonrs=Lipinski.NumHDonors(mol)
               desc_SaturatedRings = Lipinski.NumSaturatedRings(mol)   
               desc_AliphaticRings = Lipinski.NumAliphaticRings(mol) 
               desc_HAcceptors  =     Lipinski.NumHAcceptors(mol)
               desc_Heteroatoms =    Lipinski.NumHeteroatoms(mol)
               desc_Max_Partial_Charge =  Descriptors.MaxPartialCharge(mol)
               desc_FP_density =  Descriptors.FpDensityMorgan1(mol)
               desc_num_valence_electrons = Descriptors.NumValenceElectrons(mol)
               NHOH_count = Lipinski.NHOHCount(mol)
               SP3_frac = Lipinski.FractionCSP3(mol)
               SP_bonds = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[^1]')))
        #Ipc      = Descriptors.Ipc(mol)
        #HallKierAlpha= Descriptors.HallKierAlpha(mol)
        #Labute_ASA = Descriptors.LabuteASA(mol)



        #desc_molMR=Descriptors.MolMR(mol)
               row = np.array([desc_MolLogP,
                        desc_MolWt, desc_NumRotatableBonds,
                        desc_AromaticProportion,desc_Ringcount,desc_TPSA,desc_Hdonrs,desc_SaturatedRings,desc_AliphaticRings,
                        desc_HAcceptors,desc_Heteroatoms,
                        desc_Max_Partial_Charge,desc_num_valence_electrons,desc_FP_density,NHOH_count,SP3_frac,SP_bonds])
                            #,Ipc,HallKierAlpha,Labute_ASA])#,desc_num_valence_electrons])

               if i == 0:
                  baseData = row
               else:
                  baseData = np.vstack([baseData, row])
               i = i + 1

           columnNames = ["MolP","MolWt", 
                   "NumRotatableBonds", "AromaticProportion","Ring_Count","TPSA","H_donors", "Saturated_Rings","AliphaticRings","H_Acceptors","Heteroatoms","Max_Partial_Charge",
                    "valence_electrons","FP_density","NHOH_count","SP3_frac","SP_bonds"]
                  #,"Ipc","HallKierAlpha","Labute_ASA"]
           descriptors = pd.DataFrame(data=baseData, columns=columnNames)
           #st.write('Computed molecular descriptors')
           #df2 = pd.DataFrame(descriptors, index=[0])
              #descriptors1 # Skips the dummy first item
           #st.table(df2)
           trained_model= xgb.Booster()
           trained_model.load_model('models/model_xgb_95 2.bin')
           #rf_model_import = pickle.load(open('models/model_rf_93.pkl', 'rb'))
           mols = [Chem.rdmolfiles.MolFromSmiles(SMILES_string) for SMILES_string in SMILES]
           #Convert training molecules into training fingerprints
           bi = {}
           fingerprints = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo= bi, nBits=512) for m in mols]

#Convert training fingerprints into binary, and put all training binaries into arrays
           #import numpy as np 

           fingerprints_array = []
           for fingerprint in fingerprints:
             array = np.zeros((1,), dtype= int)
             DataStructs.ConvertToNumpyArray(fingerprint, array)
             fingerprints_array.append(array)

           fingerprints_array=pd.DataFrame(fingerprints_array)
           df1=pd.concat([fingerprints_array,descriptors],axis=1)
#print(df1)
           def smiles_to_sol(SMILES):
              prop=pcp.get_properties([ 'MolecularWeight'], SMILES, 'smiles')
              x = list(map(lambda x: x["CID"], prop))
              y=x[0]
   #print(y)
              x = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/%s/xml"
              data=requests.get(x % y)
    #print(data)
              html = BeautifulSoup(data.content, "xml")
              solubility = html.find(name='TOCHeading', string='Solubility')
              if solubility ==None:
                return None
              else:
                 solub=solubility.find_next_sibling('Information').find(name='String').string
                 return solub




#predict test data (MLP,XGB,RF)
#pred_mlp = mlp_model_import.predict(df1)   
           df3 = xgb.DMatrix(df1)
           pred_xgb = trained_model.predict(df3)
           pred_xgb = pred_xgb-0.18
#pred_rf = rf_model_import.predict(df1)
           mol_liter =10**pred_xgb
#mol = Chem.MolFromSmiles(SMILES)
#MolWt = Chem.Descriptors.MolWt(mol)
 
#mol_list = []
#for smiles in SMILES:
 #   mol = Chem.MolFromSmiles(smiles)
  #  mol_list.append(mol)
#MolWt = Chem.Descriptors.MolWt(mol_list)
           MolWt = descriptors["MolWt"]
           Gram_liter=(10**pred_xgb)*MolWt
           P_sol=[] ## List where I am storing the solubility data 
           for i in range(len(SMILES)): ## Dataframe which has SMILES 
             try:
    #time.sleep(1) # Sleep for 3 seconds
               sol=smiles_to_sol(SMILES[i]) ### Function to get the data from PubChem 
               P_sol.append(sol)
             except AttributeError as e:
                  sol=='No string'
                  P_sol.append(sol)
#calculate consensus
#pred_consensus=(pred_mlp+pred_xgb+pred_rf)/3
# predefined_models.get_errors(test_logS_list,pred_enseble)

# results=np.column_stack([pred_mlp,pred_xgb,pred_rf,pred_consensus])
           df_results = pd.DataFrame(SMILES, columns=['SMILES'])
           df_results["Predicted - LogS"]=pred_xgb
           df_results=df_results.round(3)
           df_results["Mol/Liter"]=mol_liter
           df_results["Gram/Liter"]=Gram_liter
           df_results["Experiment Solubility-PubChem"]=P_sol

    #df
# df_results.to_csv("results/predicted-"+test_data_name+".csv",index=False)

           st.write('Predicted LogS values')
#df_results # Skips the dummy first item
           df_results
# download=st.button('Download Results File')
# if download:
           csv = df_results.to_csv(index=False)
           b64 = base64.b64encode(csv.encode()).decode()  # some strings
           linko= f'<a href="data:file/csv;base64,{b64}" download="aquosol_predictions.csv">Download csv file</a>'
           st.markdown(linko, unsafe_allow_html=True)
 
           st.write('Computed molecular descriptors')
           #df2 = pd.DataFrame(descriptors, index=[0])
           descriptors # Skips the dummy first item
           #st.table(df2)
           #descriptors # Skips the dummy first item

      


      




      
      
     # if st.button('Prediction for input file'):
            
## Calculate molecular descriptors
      

     

            
