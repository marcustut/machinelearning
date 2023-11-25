import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SGD;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class StochasticGradientDescent {
    private static final String DATASET_PATH = "spotify_23205838.arff";
    private static final Integer NUM_FOLDS = 10;
    private static final Integer SPLIT_PCT = 80;
    private static final String[] DEPENDENT_FEATURES = { "tempo", "loudness", "liveness" };
    private static final String INDEPENDENT_FEATURE = "energy";

    public static void main(String[] args) throws Exception {
        System.out.printf("Loading dataset from %s...\n", DATASET_PATH);
        DataSource dataSource = new DataSource(DATASET_PATH);
        Instances data = dataSource.getDataSet();
        System.out.printf("Loaded %d rows.\n\n", data.size());

        // Make an indices of features that we want to keep
        ArrayList<Integer> indices = new ArrayList<Integer>();
        for (int i = 0; i < data.numAttributes(); i++)
            if (data.attribute(i).name().equals(INDEPENDENT_FEATURE)
                    || Arrays.asList(DEPENDENT_FEATURES).contains(data.attribute(i).name()))
                indices.add(i);

        // Remove all unwanted features
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndicesArray(indices.stream().mapToInt(i -> i).toArray());
        removeFilter.setInvertSelection(true);
        removeFilter.setInputFormat(data);
        Instances filteredData = Filter.useFilter(data, removeFilter);

        System.out.println("Filtered features includes:");
        // Find and set the class index
        for (int i = 0; i < filteredData.numAttributes(); i++) {
            if (filteredData.attribute(i).name().equals(INDEPENDENT_FEATURE))
                filteredData.setClassIndex(i);
            System.out.println(i + 1 + ": " + filteredData.attribute(i).name());
        }

        // Print class attribute
        System.out.println("\nClass attribute has been set to: " + filteredData.classAttribute().name());

        System.out.printf("Splitting dataset into train sets (%d%%) and test sets(%d%%)...\n", SPLIT_PCT,
                100 - SPLIT_PCT);
        int trainSize = (int) Math.round(filteredData.numInstances() * SPLIT_PCT / 100);
        Instances train = new Instances(filteredData, 0, trainSize);
        Instances test = new Instances(filteredData, trainSize, filteredData.numInstances() - trainSize);
        System.out.printf("Split complete with %d train data and %d test data.\n\n", train.size(), test.size());

        // Build the linear regression model
        SGD model = new SGD();
        model.setLossFunction(new SelectedTag(SGD.SQUAREDLOSS, SGD.TAGS_SELECTION));
        model.buildClassifier(train);
        System.out.println("\nSGD model built.");

        Evaluation eval = new Evaluation(train);
        eval.crossValidateModel(model, test, NUM_FOLDS, new Random(1));
        System.out.println(eval.toSummaryString());

        // Predict the energy
        // Instance last = filteredData.lastInstance();
        // double energy = model.classifyInstance(last);
        // System.out.println("\nPredicted energy: " + energy);
    }
}
