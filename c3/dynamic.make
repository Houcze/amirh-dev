all: main.cu libMem.so libFunc.so
	nvcc -g main.cu -o main.exe -I./ -L./ -lMem -lFunc
libMem.so: Mem.cu Mem.h
	nvcc -Xcompiler -fPIC -c Mem.cu
	nvcc -shared -o libMem.so Mem.o
	rm Mem.o
libFunc.so: Func.cu Func.h
	nvcc -Xcompiler -fPIC -c Func.cu
	nvcc -shared -o libFunc.so Func.o
	rm Func.o
main.o: main.cu
	nvcc -x cu -I./ -dc main.cu -o main.o
Func.o: Func.cu
	nvcc -x cu -I./ -dc Func.cu -o Func.o
clean:
	rm main.exe libFunc.so libMem.so