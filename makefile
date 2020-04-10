NSSolve2d: NSSolve2d.cpp
	mkdir -p out
	emcc NSSolve2d.cpp -O3 \
	-I /home/ryan/Downloads/eigen-3.3.7 \
	-s ALLOW_MEMORY_GROWTH=1 \
	-o ./out/NSSolve2d.html \
	--shell-file ./html_template/shell_minimal.html
	emcc NSSolve2d.cpp -O3 \
	-I /home/ryan/Downloads/eigen-3.3.7 \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s WASM=0 \
	-o ./out/NSSolve2dJS.html \
	--shell-file ./html_template/shell_minimalJS.html

