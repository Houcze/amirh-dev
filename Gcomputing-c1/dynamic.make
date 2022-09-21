all: main.o libMem.so Func.o
	nvcc -g main.o Func.o -o main.exe -I./ -L./ -lMem
libMem.so: Mem.cu Mem.h
	nvcc -Xcompiler -fPIC -c Mem.cu
	nvcc -shared -o libMem.so Mem.o
	rm Mem.o
%.o: main.cu Func.cu
	nvcc -x cu -I./ -dc $< -o $@
clean:
	rm main.exe libFunc.so libMem.so