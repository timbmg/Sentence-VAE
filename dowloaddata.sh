mkdir data

wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

tar -xf  simple-examples.tgz

mv simple-examples/data/ptb.train.txt data/
mv simple-examples/data/ptb.valid.txt data/
mv simple-examples/data/ptb.test.txt data/

rm -rf simple_examples
