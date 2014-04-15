FLAGS = -inline 2 -nodynlink -unsafe

DEPS = unix,bigarray

.PHONY: all clean

all:
	ocamlfind ocamlopt $(FLAGS) -S -linkpkg -package $(DEPS) -o matmult matmult.ml

clean:
	rm -f matmult
	rm -f *.cm*
	rm -f *.o
