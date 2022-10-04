CC=gcc
LINKERFLAG=-lm
# LINKERFLAG=


all: test run

test: postprocessing_test.o postprocessing.o
	@$(CC) postprocessing_test.o postprocessing.o ${LINKERFLAG} -o test

postprocessing_test.o: postprocessing_test.c postprocessing.h
	@$(CC) -c postprocessing_test.c ${LINKERFLAG} -o postprocessing_test.o

postprocessing.o: postprocessing.c postprocessing.h	
	@$(CC) -c postprocessing.c -o postprocessing.o

run: 
	@./test

clean:
	@rm -f test postprocessing_test.o postprocessing.o

