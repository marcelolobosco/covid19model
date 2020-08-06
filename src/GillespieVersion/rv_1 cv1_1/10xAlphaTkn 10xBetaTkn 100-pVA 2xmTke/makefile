V.txt: gillespie.out gillespie.cpp
	@echo "Running"
	@./gillespie.out

gillespie.out: gillespie.cpp setup.cpp
	@echo "Compiling"
	@g++ -fopenmp -Ofast gillespie.cpp -o gillespie.out

clean:
	@rm -f *.dat
	@rm -f *.out
	@rm -f *.pdf
	@clear
