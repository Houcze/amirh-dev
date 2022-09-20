all: main.cu libMem.so
	nvcc -g main.cu -o main.exe -I./ -L./ -lMem -lFunc
libMem.so: Mem.cu Mem.h
	nvcc -Xcompiler -fPIC -c Mem.cu
	nvcc -shared -o libMem.so Mem.o
	rm Mem.o
libFunc.so: Func.cu Func.h
	nvcc -Xcompiler -fPIC -c Func.cu
	nvcc -shared -o libFunc.so Func.o
	rm Func.o
clean:
	rm main.exe libFunc.so libMem.so