import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by d_d on 3/1/17.
 */
public class RandomForest {
    // Attributes' type(categorical/continuous) specification.
    private ArrayList<Boolean> typeSpecification;

    // Data CSV file delimiter.
    private String delimiter;

    // Store attributes' name if the data has a header.
    public ArrayList<String> attributesName;

    private ArrayList<DecisionTree> randomForest;

    private Entries trainData;

    private Entries testData;

    private ArrayList<Boolean> chosenAttributes;

    public double trainSubsetFraction;

    public int attrSubspaceNum;


    /**
     * A utility that facilitates the counting process in a hash map for certain key.
     * Same as in python Collections.Counter().
     * @param hashMap The hash map that needs to be updated.
     * @param key The key that needs to be updated.
     * @return The updated hash map.
     */
    private Map<String, Integer> Counter(Map<String, Integer> hashMap, String key) {
        Map<String, Integer> temp = new HashMap<>(hashMap);
        if (temp.containsKey(key)) {
            int count = temp.get(key);
            temp.put(key, ++ count);
        } else {
            temp.put(key, 1);
        }
        return temp;
    }

    /**
     * A utility function to read a CSV as a List of String Arrays, each element is a row.
     * @param filePath The CSV filepath.
     * @return The rows raw data.
     * @throws IOException In case of IOException.
     */
    public List<String[]> readCSV(String filePath, boolean header) throws IOException {
        BufferedReader fileReader = new BufferedReader(new FileReader(filePath));
        String line;
        List<String[]> entries = new ArrayList<>();

        // Process header;
        if (header) {
            this.attributesName = new ArrayList<>();
            line = fileReader.readLine();

            this.attributesName.addAll(Arrays.asList(line.split(this.delimiter)));

            this.attrSubspaceNum = (int) Math.sqrt(this.attributesName.size());

        }
        while ((line = fileReader.readLine()) != null) {

            String[] tokens = line.split(this.delimiter);

            if (tokens.length > 0) {
                entries.add(tokens);
            }
        }
        return entries;
    }

    /**
     * A utility function for loadData function, to add data to DecisionTree properties.
     * @param training Indicate if the data is training data.
     * @param entries The entries needs to be filled in.
     */
    private void loadDataUtil(boolean training, List<String[]> entries) {
        for (String[] s: entries) {
            int i;

            Entry newEntry = new Entry();
            for (i = 0; i < s.length - 1; i ++) {
                newEntry.attributes.add(new CellData(s[i], this.typeSpecification.get(i)));
            }

            // The last column is as default the label.
            newEntry.label = s[i];

            if (training) {
                this.trainData.entries.add(newEntry);
            } else {
                this.testData.entries.add(newEntry);
            }
        }
    }

    /**
     * An alternate public method for user to load entries data instead of from file.
     * @param training Indicate if the file is training data.
     * @param entries The entries needs to be filled in.
     * @throws IOException In case of IOException.
     */
    public void loadData(boolean training, ArrayList<String[]> entries) throws IOException {
        loadDataUtil(training, entries);
    }


    public RandomForest(ArrayList<Boolean> typeSpecification, ArrayList<Boolean> choosenAttributes, String delimiter) {
        this.randomForest = new ArrayList<>();
        this.trainData = new Entries();
        this.testData = new Entries();
        this.typeSpecification = typeSpecification;
        this.chosenAttributes = choosenAttributes;
        this.delimiter = delimiter;
        this.trainSubsetFraction = 2.0 / 3.0;

        this.attributesName = null;
    }


    public void initialize(int treesNumber) {
        for (int i = 0; i < treesNumber; i ++) {
            DecisionTree newDecisionTree = new DecisionTree(this.typeSpecification, this.chosenAttributes, this.delimiter, true);
            this.randomForest.add(newDecisionTree);
        }
    }

    public void startTraining() {
        int trainSubsetSize = (int) (this.trainData.entries.size() * this.trainSubsetFraction);

        for (DecisionTree dt: this.randomForest) {

            System.out.println("Tree " + this.randomForest.indexOf(dt) + ":");

            ArrayList<Integer> trainIndexes = new ArrayList<>();

            for (int i = 0; i < trainSubsetSize; i ++) {
                Integer index = (int) (Math.random() * (this.trainData.entries.size()));

                while (trainIndexes.contains(index)) {
                    index = (int) (Math.random() * (this.trainData.entries.size()));
                }

                trainIndexes.add(index);

                dt.trainData.entries.add(this.trainData.entries.get(index));
            }

            dt.attrSubspaceNum = this.attrSubspaceNum;

            dt.attributesName = this.attributesName;

            dt.startTraining();

            dt.preorderTraversePrint(dt.start, dt.root, -1, false, true);

            System.out.println("\n\n");
        }
    }


    public double startTesting() {
        double correct = 0;
        double all = 0;
        for (Entry e: this.testData.entries) {

            Map<String, Integer> predictedLabels = new HashMap<>();

            for (DecisionTree dt : this.randomForest) {
                String predictedLabel = dt.startTesting(e);
                predictedLabels = Counter(predictedLabels, predictedLabel);
                System.out.print(predictedLabel + "\t");
            }

            String finalLabel = Collections.max(predictedLabels.entrySet(), Map.Entry.comparingByValue()).getKey();

            System.out.print("\nFinal: " + finalLabel + ", True: " + e.label + "\n");

            if (finalLabel.equals(e.label)) {
                correct ++;
            } else {
                System.out.print("Miss classifying [ ");
                for (CellData d: e.attributes) {
                    System.out.print(d.value + ", ");
                }
                System.out.print(e.label + "]\tas [" + finalLabel + "]\n");
            }

            System.out.println();

            all ++;
        }
        double accuracy = correct / all;
        System.out.println("Accuracy: " + accuracy);
        return accuracy;
    }
}
