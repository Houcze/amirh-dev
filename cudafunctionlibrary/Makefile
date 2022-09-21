all: main.cu libfunc.so
	nvcc -g main.cu -o main.exe -I./ -L./ -lfunc
libfunc.so: func.cu func.h
	nvcc -Xcompiler -fPIC -c func.cu
	nvcc -shared -o libfunc.so func.o
	rm func.o
clean:
	rm main.exe libfunc.so