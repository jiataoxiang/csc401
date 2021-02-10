rm ./output/compare*

grep "body" test.json >> ./output/compare1
grep "body" ./output/sample_out.json >> ./output/compare2

diff ./output/compare1 ./output/compare2
