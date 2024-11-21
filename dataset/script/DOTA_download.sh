#! /bin/bash
conda activate base
conda install gdown -y

ROOT=$PWD

cd $ROOT
mkdir train
cd train
# train image
gdown --folder -q https://drive.google.com/drive/folders/1MvSH7sNaY4p4lhwAU_BG3y7zth6-rtrD?usp=drive_link
# train label-v1.0
gdown --folder -q https://drive.google.com/drive/folders/1cCU7Mxs2YqQ26UBJcQklB1ZI6bv5fCt4?usp=drive_link
# train label-v1.5
# gdown --folder -q https://drive.google.com/drive/folders/1_ObTxq-IhRL-N4Ci9HTR7upxNTBTzaRL?usp=drive_link

cd $ROOT
mkdir val
cd val
# val image
gdown --folder -q https://drive.google.com/drive/folders/1RV7Z5MM4nJJJPUs6m9wsxDOJxX6HmQqZ?usp=drive_link
# val label-v1.0
gdown --folder -q https://drive.google.com/drive/folders/1nySK57mC4mEgblwCXW4BRyK2voly13LM?usp=drive_link
# val label -v1.5
# gdown --folder -q https://drive.google.com/drive/folders/1_ncfVzGon8br8ZKjo-HVzkL_ptEzwj8L?usp=drive_link

cd $ROOT
mkdir test
cd test
# test image
gdown --fuzzy -q https://drive.google.com/file/d/1fwiTNqRRen09E-O9VSpcMV2e6_d4GGVK/view?usp=drive_link
gdown --fuzzy -q https://drive.google.com/file/d/1wTwmxvPVujh1I6mCMreoKURxCUI8f-qv/view?usp=drive_link
# test label
gdown --fuzzy -q https://drive.google.com/file/d/1nQokIxSy3DEHImJribSCODTRkWlPJLE3/view?usp=drive_link
