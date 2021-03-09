mkdir -p output

rm ./output/compare*

python3 a1_preproc.py 1004236613 -o test.json --a1_dir ./

grep "body" test.json >> ./output/compare1
grep "body" ./output/sample_out.json >> ./output/compare2

diff ./output/compare1 ./output/compare2
