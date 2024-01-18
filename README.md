# IrOLED_scripts_data
Project folder with scripts of models, csv files of data, xyz files of chemical dataset, and output files

## Contents:

environment.yml: file to recreate the conda environment used to execute the project

Use following code to recreate environment:

    conda env create -f environment.yml
    conda activate IrOLED
    pip install -e ./megnet-master/.


### Scripts:
mlp_model.py: script to run MLP model with ligand property features as input

lig_MEGNet_model.py: script to run lig-MEGNet model with ligand XYZ files as input or ligand SMILES as input

mlp_load_predict.py: script to load a saved mlp model and the values of a test ligand's property features to generate predictions

ligMEG_load_predict.py: script to load a saved lig-MEGNet model and the XYZ files of the test complex's ligands to generate predictions


### Models:
new_mlp_model.hdf5: saved model from a "mlp_model.py" run

ligsMEG_model.hdf5: saved model from a "lig_MEGNet_model.py" run with ligand XYZ files as input

ligsMEG_smiles_input_model.hdf5: saved model from a "lig_MEGNet_model.py" run with ligand SMILES as input


### Folders:
#### csv_files:
  IrCNNN_smiles_wts.csv: numbered list of IrCNNN complexes, CN ligand, NN ligands, respective SMILES, and weights used in modeling 
  
  lig_prop_data.csv: list of the ligand (both CN and NN) property features of the high-throughput dataset used in the mlp model  
  
  Spectra_46_intensities.csv: list of the normalized spectral intensities of the IrCNNN complexes
  
  CN110_NN6.csv: property feature values for CN110-NN6 complex
  
  CN110_NN40.csv: property feature values for CN110-NN40 complex
  
  CN111_NN6.csv: property feature values for CN111-NN6 complex
  
  CN111_NN40.csv: property feature values for CN111-NN40 complex


#### out_files:
  mlp_result.out: output file of a "mlp_model.py" run

  lig_MEG_result.out: output file of a "lig_MEGNet_model.py" run with ligand XYZ files as input

  lig_MEG_smiles_input_result.out: output file of a "lig_MEGNet_model.py" run with ligand SMILES as input

  [complex]_[model]_pred.out: output file with prediction for the complex using the model 


#### ligs_xyz:
  xyz files of all the CN and NN ligands from the high-throughput dataset as well as the new CN ligands, CN110 and CN 111


#### megnet-master:
  local version of modified MEGNet code used in the scripts above


