
# download all semeval files

wget -O datasets/en-semeval.zip  https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_eng.zip
wget -O datasets/de-semeval.zip https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_ger.zip
wget -O datasets/ln-semeval.zip https://zenodo.org/record/3734089/files/semeval2020_ulscd_lat.zip?download=1
wget -O datasets/sw-semeval.zip https://zenodo.org/record/3730550/files/semeval2020_ulscd_swe.zip?download=1

# unzip and move en-semeval

unzip datasets/en-semeval.zip 

mv semeval2020_ulscd_eng/corpus1/lemma/ccoha1.txt.gz datasets/en-semeval/c1.txt.gz
mv semeval2020_ulscd_eng/corpus2/lemma/ccoha2.txt.gz datasets/en-semeval/c2.txt.gz
mv semeval2020_ulscd_eng/truth/graded.txt datasets/en-semeval/truth.tsv
mv semeval2020_ulscd_eng/targets.txt datasets/en-semeval/targets.tsv

rm -r semeval2020_ulscd_eng/

gunzip datasets/en-semeval/c1.txt.gz 
gunzip datasets/en-semeval/c2.txt.gz 

# unzip and move de-semeval

unzip datasets/de-semeval.zip 

mv semeval2020_ulscd_ger/corpus1/lemma/dta.txt.gz datasets/de-semeval/c1.txt.gz
mv semeval2020_ulscd_ger/corpus2/lemma/bznd.txt.gz datasets/de-semeval/c2.txt.gz
mv semeval2020_ulscd_ger/truth/graded.txt datasets/de-semeval/truth.tsv
mv semeval2020_ulscd_ger/targets.txt datasets/de-semeval/targets.tsv

rm -rf semeval2020_ulscd_ger/

gunzip datasets/de-semeval/c1.txt.gz 
gunzip datasets/de-semeval/c2.txt.gz 

# unzip and move ln-semeval

unzip datasets/ln-semeval.zip 

mv semeval2020_ulscd_lat/corpus1/lemma/LatinISE1.txt.gz datasets/ln-semeval/c1.txt.gz
mv semeval2020_ulscd_lat/corpus2/lemma/LatinISE2.txt.gz datasets/ln-semeval/c2.txt.gz
mv semeval2020_ulscd_lat/truth/graded.txt datasets/ln-semeval/truth.tsv
mv semeval2020_ulscd_lat/targets.txt datasets/ln-semeval/targets.tsv

rm -rf semeval2020_ulscd_lat/

gunzip datasets/ln-semeval/c1.txt.gz 
gunzip datasets/ln-semeval/c2.txt.gz 

# unzip and move sw-semeval

unzip datasets/sw-semeval.zip 

mv semeval2020_ulscd_swe/corpus1/lemma/kubhist2a.txt.gz datasets/sw-semeval/c1.txt.gz
mv semeval2020_ulscd_swe/corpus2/lemma/kubhist2b.txt.gz datasets/sw-semeval/c2.txt.gz
mv semeval2020_ulscd_swe/truth/graded.txt datasets/sw-semeval/truth.tsv
mv semeval2020_ulscd_swe/targets.txt datasets/sw-semeval/targets.tsv

rm -rf semeval2020_ulscd_swe/

gunzip datasets/sw-semeval/c1.txt.gz 
gunzip datasets/sw-semeval/c2.txt.gz 


