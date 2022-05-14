C = hipcc
CFLAGS = -std=c++11

all: transpose 

transpose: transpose.o mat.o 
	$(C) $(CFLAGS) -o transpose.exe transpose.o mat.o 
transpose.o: transpose.cpp
	$(C) $(CFLAGS) -c transpose.cpp -o transpose.o
mat.o: mat.cpp
	$(C) $(CFLAGS) -c mat.cpp -o mat.o

clean:
	rm -f transpose.exe *.dat *.o


