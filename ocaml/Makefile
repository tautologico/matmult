all:
	ocamlfind ocamlopt -thread -linkpkg -package core,core_bench -o matmult matmult.ml

clean:
	rm -f matmult
	rm -f *.cm*
	rm -f *.o
