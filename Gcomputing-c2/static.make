all: main.cu libMem.a libFunc.a
	nvcc -g main.cu -o main.exe -I./ -L./ -lMem -lFunc
	rm libMem.a libFunc.a
libMem.a: Mem.cu Mem.h
	nvcc -c Mem.cu
	ar rcs libMem.a Mem.o
	rm Mem.o
libFunc.a: Func.cu Func.h
	nvcc -c Func.cu
	ar rcs libFunc.a Func.o
	rm Func.o
clean:
	rm main.exe