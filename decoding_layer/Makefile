CC=gcc
LINKERFLAG = -lm

all: test run 

test: test_decoding.c decoding 
	@$(CC) test_decoding.c decoding ${LINKERFLAG} -o test 

decoding: ../inference_gvsoc/decoding.c ../inference_gvsoc/decoding.h 
	@$(CC) -c ../inference_gvsoc/decoding.c -o decoding

run: 
	@./test

clean: 
	@rm -f test decoding