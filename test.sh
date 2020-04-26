cat datasets/de-surel/c1_*.txt >> datasets/de-surel/c1_full.txt
sed -n '0~8p' datasets/de-surel/c1_full.txt >> datasets/de-surel/c1.txt # in line with Wind of Change (Schlechtweg et al., 2019)
rm datasets/de-surel/c1_full.txt
