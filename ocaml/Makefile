all:
	ocamlfind ocamlopt -inline 2 -S -linkpkg -package unix,bigarray -o matmult matmult.ml

clean:
	rm -f matmult
	rm -f *.cm*
	rm -f *.o
