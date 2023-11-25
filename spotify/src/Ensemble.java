import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.RandomizableParallelIteratedSingleClassifierEnhancer;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.RandomSubSpace;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

class Coord {
    public Integer x;
    public Double y;

    public Coord(Integer x, Double y) {
        this.x = x;
        this.y = y;
    }

    public String toString() {
        return String.format("(%d, %.3f)", x, y);
    }
};

class EnsembleResult {
    public ArrayList<Coord> dtree;
    public ArrayList<Coord> nn;
    public ArrayList<Coord> knn;

    public EnsembleResult() {
        this.dtree = new ArrayList<Coord>();
        this.nn = new ArrayList<Coord>();
        this.knn = new ArrayList<Coord>();
    }

    private static String coordsToString(ArrayList<Coord> coords) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < coords.size(); i++) {
            sb.append(coords.get(i).toString());
            if (i < coords.size() - 1)
                sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }

    public String toString() {

        return String.format("dtree: %s\nnn: %s\nknn: %s\n", EnsembleResult.coordsToString(dtree),
                EnsembleResult.coordsToString(nn),
                EnsembleResult.coordsToString(knn));
    }
};

public class Ensemble {
    private static final String DATASET_PATH = "spotify_23205838.arff";
    private static final Integer NUM_FOLDS = 10;
    private static final Integer SPLIT_PCT = 80;
    private static final AbstractClassifier[] CLASSIFIERS = {
            new J48(),
            new MultilayerPerceptron(),
            new IBk(1), // k = 1
    };

    private interface EnsembleBuilder {
        public RandomizableParallelIteratedSingleClassifierEnhancer build(Integer ensembleSize,
                AbstractClassifier classifier) throws Exception;
    }

    private interface EnsembleEvaluator {
        public Evaluation evaluate(RandomizableParallelIteratedSingleClassifierEnhancer ensemble)
                throws Exception;

    }

    public static void main(String[] args) throws Exception {
        System.out.printf("Loading dataset from %s...\n", DATASET_PATH);
        DataSource dataSource = new DataSource(DATASET_PATH);
        Instances data = dataSource.getDataSet();
        System.out.printf("Loaded %d rows.\n\n", data.size());

        data.setClassIndex(data.numAttributes() - 1);
        System.out.printf("Class attribute set to %s\n\n", data.classAttribute().name());

        System.out.printf("Splitting dataset into train sets (%d%%) and test sets(%d%%)...\n", SPLIT_PCT,
                100 - SPLIT_PCT);
        int trainSize = (int) Math.round(data.numInstances() * SPLIT_PCT / 100);
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);
        System.out.printf("Split complete with %d train data and %d test data.\n\n", train.size(), test.size());

        Integer[] ensembleSizes = { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 };
        System.out.printf("ensembleSizes: %s...\n\n", Arrays.toString(ensembleSizes));

        EnsembleEvaluator evaluator = (ensemble) -> {
            Evaluation eval;

            if (ensemble.getClassifier().getClass().getName().contains("MultiLayerPerceptron")) {
                System.out.println("Creating Evaluation for MLP...");
                eval = new Evaluation(train);
                System.out.println("Evaluating the model...");
                eval.evaluateModel(ensemble, test);
            } else {
                System.out.println("Creating Evaluation...");
                eval = new Evaluation(data);
                System.out.println("Evaluating the model...");
                eval.crossValidateModel(ensemble, data, NUM_FOLDS, new Random(1));
            }

            return eval;
        };

        Ensemble main = new Ensemble();
        EnsembleResult baggingResult = main.run(ensembleSizes, (bagSize, classifier) -> {
            Bagging ensemble = new Bagging();
            ensemble.setNumIterations(bagSize);
            ensemble.setClassifier(classifier);

            System.out.printf("Running ensemble %s [bagSize=%d] with classifier %s\n",
                    ensemble.getClass().getName(), bagSize,
                    classifier.getClass().getName());

            if (ensemble.getClassifier().getClass().getName().contains("MultiLayerPerceptron")) {
                MultilayerPerceptron _classifier = (MultilayerPerceptron) ensemble.getClassifier();
                _classifier.setTrainingTime(1);
                ensemble.buildClassifier(train);
            } else
                ensemble.buildClassifier(data);

            return ensemble;
        }, evaluator);
        System.out.println("Result (ensemble with bagging): ");
        System.out.println(baggingResult);

        EnsembleResult randomSubspacingResult = main.run(ensembleSizes, (subSpaceSize, classifier) -> {
            RandomSubSpace ensemble = new RandomSubSpace();
            ensemble.setSubSpaceSize(subSpaceSize);
            ensemble.setClassifier(classifier);

            System.out.printf("Running ensemble %s [subSpaceSize=%d] with classifier %s\n",
                    ensemble.getClass().getName(), subSpaceSize,
                    classifier.getClass().getName());

            if (ensemble.getClassifier().getClass().getName().contains("MultiLayerPerceptron"))
                ensemble.buildClassifier(train);
            else
                ensemble.buildClassifier(data);
            return ensemble;
        }, evaluator);
        System.out.println("Result (ensemble with random subspacing): ");
        System.out.println(randomSubspacingResult);
    }

    private EnsembleResult run(Integer[] ensembleSizes, EnsembleBuilder builder, EnsembleEvaluator evaluator)
            throws Exception {
        EnsembleResult result = new EnsembleResult();

        for (AbstractClassifier classifier : CLASSIFIERS) {
            for (Integer ensembleSize : ensembleSizes) {
                long start = System.currentTimeMillis();
                RandomizableParallelIteratedSingleClassifierEnhancer ensemble = builder.build(ensembleSize, classifier);
                long end = System.currentTimeMillis();
                System.out.printf("Took %.2fs to build ensemble.\n", (end - start) / 1000.0);

                System.out.printf("Evaluating model...\n");
                start = System.currentTimeMillis();
                Evaluation eval = evaluator.evaluate(ensemble);
                end = System.currentTimeMillis();
                System.out.printf("Took %.2fs to complete evaluation.\n", (end - start) / 1000.0);
                System.out.printf("Weighted F1-measure: %.3f\n\n", eval.weightedFMeasure());

                if (classifier.getClass().getName().contains("J48"))
                    result.dtree.add(new Coord(ensembleSize, eval.weightedFMeasure()));
                else if (classifier.getClass().getName().contains("MultilayerPerceptron"))
                    result.nn.add(new Coord(ensembleSize, eval.weightedFMeasure()));
                else if (classifier.getClass().getName().contains("IBk"))
                    result.knn.add(new Coord(ensembleSize, eval.weightedFMeasure()));
            }
        }

        return result;
    }
}
