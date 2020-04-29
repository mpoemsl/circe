
# download de-durel text files

wget -O datasets/de-durel/c1.txt.gz https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/wocc/dta18.txt.gz
wget -O datasets/de-durel/c2.txt.gz https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/wocc/dta19.txt.gz

# download de-surel text files

wget -O datasets/de-surel/c1_1.txt.gz https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/wocc/sdewac_1.txt.gz
wget -O datasets/de-surel/c1_2.txt.gz https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/wocc/sdewac_2.txt.gz
wget -O datasets/de-surel/c1_3.txt.gz https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/wocc/sdewac_3.txt.gz
wget -O datasets/de-surel/c2.txt.gz https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/wocc/cook.txt.gz

# unzip de-durel

gunzip datasets/de-durel/c1.txt.gz
gunzip datasets/de-durel/c2.txt.gz

# unzip and preprocess de-surel 

gunzip datasets/de-surel/c1_1.txt.gz
gunzip datasets/de-surel/c1_2.txt.gz
gunzip datasets/de-surel/c1_3.txt.gz
gunzip datasets/de-surel/c2.txt.gz

cat datasets/de-surel/c1_*.txt >> datasets/de-surel/c1_full.txt
sed -n '0~8p' datasets/de-surel/c1_full.txt >> datasets/de-surel/c1.txt # in line with Wind of Change (Schlechtweg et al., 2019)
rm datasets/de-surel/c1_*.txt
