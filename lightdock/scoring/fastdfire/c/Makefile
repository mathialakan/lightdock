#
#LDSHARED=$(shell which nvcc)
#LDSHARED=$(shell which pgcc)
#CC=$(shell which nvcc)
#CC=$(shell which pgcc)


all: 
	CC=$(shell which gcc) LDSHARED=$(shell which gcc) OPT="-g -O3" python3 setup.py build_ext --inplace
	python3 setup.py clean

build:

clean:
	rm -rf *.so
