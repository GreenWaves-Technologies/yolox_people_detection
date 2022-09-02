CC=gcc

test: test_slicing.c slicing.o
	$(CC)  test_slicing.c slicing.o -o test

slicing.o: ../inference_gvsoc/slicing.h ../inference_gvsoc/slicing.c
	$(CC) -c ../inference_gvsoc/slicing.c 

run: 
	./test 

clean: 
	rm -f test slicing.o 