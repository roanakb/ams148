C = hipcc
CFLAGS = -std=c++11

all: matvec 

matvec: matvec.o mat.o 
	$(C) $(CFLAGS) -o matvec.exe matvec.o mat.o 
matvec.o: matvec.cpp
	$(C) $(CFLAGS) -c matvec.cpp -o matvec.o
mat.o: mat.cpp
	$(C) $(CFLAGS) -c mat.cpp -o mat.o

clean:
	rm -f matvec.exe *.dat *.o


