all: m.cu libOps.so
	nvcc -g m.cu -o m.exe -I./ -L./ -lOps
libOps.so: Ops.cu Ops.h
	nvcc -Xcompiler -fPIC -c Ops.cu
	nvcc -shared -o libOps.so Ops.o
	rm Ops.o
clean:
	rm m.exe libOps.so
run:
	./m.exe