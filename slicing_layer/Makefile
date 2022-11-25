CC=gcc

test: test_slicing.c slicing.o
	$(CC)  test_slicing.c slicing.o -o test

slicing.o: ../inference_gvsoc_240x320_int/slicing.h ../inference_gvsoc_240x320_int/slicing.c
	$(CC) -c ../inference_gvsoc_240x320_int/slicing.c 

run: 
	./test 

clean: 
	rm -f test slicing.o 