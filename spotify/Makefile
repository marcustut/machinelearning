CLASSPATH=lib/weka.jar

compile:
	javac -cp $(CLASSPATH) src/* -d bin

run-ensemble: compile
	java -cp $(CLASSPATH):bin Ensemble

run-linear-regression: compile
	java -cp $(CLASSPATH):bin LinearRegression

run-sgd: compile
	java -cp $(CLASSPATH):bin StochasticGradientDescent