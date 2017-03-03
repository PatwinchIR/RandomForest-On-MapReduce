import javafx.util.Pair;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by d_d on 2/21/17.
 * This class is for Training and Testing purpose;
 * This class of Decision Tree can only deal with continuous attributes value and discrete output labels.
 * This class can only support CSV fomatted data file. Training and Test dataset format should be:
 * ||============================================||
 * ||     attr1,attr2,attr3,...,attrn,label      ||
 * ||============================================||
 * For continuous output, Please refer to regression tree, decision tree is more like a classifier.
 *
 * ===================================================
 * Updated by d_d on 3/1/17.
 *
 * This class of Decision Tree can deal with [continuous, categorical or a mixture of both]
 * attributes value and discrete output labels.
 * This class can only support CSV fomatted data file. Training and Test dataset format should be:
 * ||============================================||
 * ||     attr1,attr2,attr3,...,attrn,label      ||
 * ||============================================||
 * The delimiter can be specified when create new instance of this class.
 * The attributes type needs to be specified beforehand, true if the type is categorical, false otherwise.
 * For continuous output, Please refer to regression tree, decision tree is more like a classifier.
 *
 * ===================================================
 * Updated by d_d on 3/3/17.
 *
 * This class now support random forest.
 *
 */
public class DecisionTree {
    /**
     * Decision Tree Constructor.
     * Initialise training dataset and testing dataset.
     * @param typeSpecification Attributes' type(categorical/continuous) specification.
     * @param chosenAttributes A boolean array indicates the attributes that user choose to use/ignore.
     * @param delimiter Data CSV file delimiter.
     * @param inRandomForest For RandomForest use indicator.
     */
    public DecisionTree(ArrayList<Boolean> typeSpecification, ArrayList<Boolean> chosenAttributes, String delimiter, boolean inRandomForest) {
        this.trainData = new Entries();
        this.testData = new Entries();
        this.typeSpecification = typeSpecification;
        this.chosenAttributes = chosenAttributes;
        this.delimiter = delimiter;

        this.inRandomForest = inRandomForest;
        this.attributesName = null;
    }

    // A boolean array indicates the attributes that user choose to use/ignore.
    private ArrayList<Boolean> chosenAttributes;

    // Store attributes' name if the data has a header.
    public ArrayList<String> attributesName;

    // For RandomForest use indicator.
    private boolean inRandomForest;

    // Data CSV file delimiter.
    private String delimiter;

    // Attributes' type(categorical/continuous) specification.
    private ArrayList<Boolean> typeSpecification;

    // Training Data.
    public Entries trainData;

    // Testing Data.
    public Entries testData;

    // Decision Tree's root node.
    public Node root;

    // For visualization/output purpose.
    public Node start;

    // The confusion matrix.
    private Map<Pair<String, String>, Integer> confusionMatrix;

    // Indicates the Random subspace in Random Forest.
    public int attrSubspaceNum;

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
     * A public method for user to load data for DecisionTree class.
     * NOTICE: Training data and test data need to be loaded seperately, label is as default the last column
     *         in the dataset.
     * @param training  Indicate if the file is training data.
     * @param filePath  Indicate the filepath.
     * @throws IOException In case of IOException.
     */
    public void loadData(boolean training, String filePath, boolean header) throws IOException {
        List<String[]> entries = readCSV(filePath, header);
        loadDataUtil(training, entries);
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

    /**
     * The main ID3 recursive function. The pseudocode can be found at:
     * https://www.cs.swarthmore.edu/~meeden/cs63/f05/id3.html
     * @param examples  The examples for next splitting.
     * @param attributes    The attributes for next splitting. (Remaining Attributes.)
     * @return  The root node of the DecisionTree.
     */
    private Node ID3(Entries examples, ArrayList<Integer> attributes){
        Node node = new Node(examples, attributes, this.typeSpecification, this.chosenAttributes, this.inRandomForest, this.attrSubspaceNum);

        // If current node is already consistent with examples, return.
        if (node.isConsistent) {
            return node;

        } else {
            // If there's no longer attributes, no need to continue.
            if (attributes.size() == 0) {

                // There's no attributes to continue splitting though current node is not consistent, then
                // take a majority vote for this leaf node's label.
                node.label = Collections.max(node.labelsCount.entrySet(), Map.Entry.comparingByValue()).getKey();
                return node;

            }

            Entries newExamplesLeft = new Entries();
            Entries newExampleRight = new Entries();

            if (!this.typeSpecification.get(node.bestAttribute)) {
                // Sort examples according to best splitting attribute, for splitting dataset later.
                Collections.sort(examples.entries, new Comparator<Entry>() {
                    @Override
                    public int compare(Entry o1, Entry o2) {
                        return new CellData().compare(o1.attributes.get(node.bestAttribute),
                                o2.attributes.get(node.bestAttribute));
                    }
                });

                // Split dataset according to decision of the best splitting attribute.
                for (int i = 0; i < examples.entries.size(); i++) {
                    Entry entryTemp = examples.entries.get(i);
                    if ((Double) entryTemp.attributes.get(node.bestAttribute).value <= ((Double) node.decision.value)) {
                        newExamplesLeft.entries.add(entryTemp);
                    } else {
                        newExampleRight.entries.add(entryTemp);
                    }
                }
            } else {
                // Split dataset according to decision of the best splitting attribute.
                for (int i = 0; i < examples.entries.size(); i++) {
                    Entry entryTemp = examples.entries.get(i);
                    if ((entryTemp.attributes.get(node.bestAttribute).value).equals(node.decision.value)) {
                        newExamplesLeft.entries.add(entryTemp);
                    } else {
                        newExampleRight.entries.add(entryTemp);
                    }
                }
            }

            // Generating remaining attributes.
            ArrayList<Integer> newAttributes = new ArrayList<>(attributes);
            newAttributes.remove(Integer.valueOf(node.bestAttribute));

            // If the dataset after splitting is not empty, then branching and grow the tree. Else end growing.
            if (newExamplesLeft.entries.size() != 0) {
                node.left = ID3(newExamplesLeft, newAttributes);
            } else {
                node.left = new Node();
            }
            if (newExampleRight.entries.size() != 0) {
                node.right = ID3(newExampleRight, newAttributes);
            } else {
                node.right = new Node();
            }

            return node;
        }
    }

    /**
     * Using preorder traversal to print the ouput for required visualizaiton.
     * Because the desired output requires to show the splitting attribute on the child node, along with the
     * decision boundary, and to use "<" or ">=" to represent left and right child respectively.
     * @param parent Indicate child's desired output for attribute and decision boundary.
     * @param node  Indicate current node.
     * @param depth Indicate the depth of current node.
     * @param right Indicate if current node is a right child of its parent, to determine "<" and ">=".
     * @param start Indicate if it's the starting node(before root).
     */
    public void preorderTraversePrint(Node parent, Node node, Integer depth, boolean right, boolean start) {
        if (node == null) {
            return;
        }
        for (int i = 0; i < depth; i ++){
            System.out.print("\t");
        }

        // Decision relation indicator.
        String relation;

        if (!start){
            if (!this.typeSpecification.get(parent.bestAttribute)) { // Continuous attribute.
                if (!right) {
                    relation = " <= ";
                } else {
                    relation = " > ";
                }
            } else { // Categorical attribute.
                if (!right) {
                    relation = " == ";
                } else {
                    relation = " != ";
                }
            }
            if (this.attributesName == null) {
                System.out.print("|Attr" + parent.bestAttribute + relation + parent.decision.value + "|Entropy: " + node.entropy + " : " + node.label + " ");
            } else { // If the data CSV has a header.
                System.out.print("|" + this.attributesName.get(parent.bestAttribute) + relation + parent.decision.value + "|Entropy: " + node.entropy + " : " + node.label + " ");
            }
        }
        Iterator it = node.labelsCount.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry pair = (Map.Entry)it.next();
            String label = (String) pair.getKey();
            Integer count = (Integer) pair.getValue();
            System.out.print("[" + label + " : " + count + "]");
        }
        System.out.print("\n");
        preorderTraversePrint(node, node.left, depth + 1, false, false);
        preorderTraversePrint(node, node.right, depth + 1, true, false);
    }

    /**
     * Using current test entry to traverse the built tree and return predicted label.
     * @param entry Current test entry.
     * @param node  The decision tree's root node.
     * @return  The current test entry's predicted label.
     */
    private String getPrediction(Entry entry, Node node) {
        if (node.left == null && node.right == null)
            return node.label;
        if (!this.typeSpecification.get(node.bestAttribute)) {
            if (((Double) entry.attributes.get(node.bestAttribute).value) <= ((Double) node.decision.value)) {
                return getPrediction(entry, node.left);
            } else {
                return getPrediction(entry, node.right);
            }
        } else {
            if ((entry.attributes.get(node.bestAttribute).value).equals(node.decision.value)) {
                return getPrediction(entry, node.left);
            } else {
                return getPrediction(entry, node.right);
            }
        }
    }

    /**
     * A utility function to print n whitespaces.
     * @param n The number of whitespaces to print.
     */
    private void printWhitespaces(int n) {
        for (int i = 0; i < n; i ++){
            System.out.print(" ");
        }
    }

    /**
     * A utility function to print Confusion Matrix delimiter.
     */
    private void printDelimeter() {
        System.out.print("|");
    }

    /**
     * A utility function to print line seperator.
     * @param n The number of columns.
     * @param size The width of per column.
     */
    private void printLine(int n, int size) {
        printDelimeter();
        for (int i = 0; i < n * (size + 1) + size; i ++) {
            System.out.print("=");
        }
        printDelimeter();
        System.out.println();
    }

    /**
     * A utility function to print Matrix head.
     * @param n The number of columns.
     * @param size The width of per column.
     */
    private void printMatrixHead(int n, int size) {
        int width = n * (size + 1) + size;
        String head = "Confusion Matrix";
        printWhitespaces((width - head.length()) / 2);
        System.out.println(head);
        printLine(n, size);
    }

    /**
     * A utility to print confusion matrix.
     */
    public void confusionMatrixPrint() {
        Map<String, Integer> labelsCount = this.root.labelsCount;
        List<String> labels = new ArrayList<>();

        int longest = 0;
        Iterator it = labelsCount.entrySet().iterator();
        while (it.hasNext()) {
            String al = (String) ((Map.Entry) it.next()).getKey();
            longest = al.length() > longest ? al.length() : longest;
            if (!labels.contains(al)) {
                labels.add(al);
            }
        }
        printMatrixHead(longest, labels.size());
        printDelimeter();
        printWhitespaces(longest);
        printDelimeter();
        for (String label: labels) {
            printWhitespaces(longest - label.length());
            System.out.print(label);
            printDelimeter();
        }
        System.out.print("\n");
        printLine(longest, labels.size());
        for (int i = 0; i < labels.size(); i ++) {
            String al = labels.get(i);
            printDelimeter();
            printWhitespaces(longest - al.length());
            System.out.print(al);
            printDelimeter();
            for (String label: labels) {
                Integer count = this.confusionMatrix.get(new Pair<>(al, label));
                String num = count == null ? "0" : String.valueOf(count);
                printWhitespaces(longest - num.length());
                System.out.print(num);
                printDelimeter();
            }
            System.out.print("\n");
            printLine(longest, labels.size());
        }
    }

    /**
     * A utility to update confusion matrix.
     * @param pair The <actual label, predict label> pair.
     */
    private void updateConfusionMatrix(Pair<String, String> pair) {
        if (this.confusionMatrix.containsKey(pair)) {
            int count = this.confusionMatrix.get(pair);
            this.confusionMatrix.put(pair, ++ count);
        } else {
            this.confusionMatrix.put(pair, 1);
        }
    }


    /**
     * Funtion to start building the tree.
     */
    public void startTraining() {
        ArrayList<Integer> attributes = new ArrayList<>();

        // The attributes index array. To indicate the remaining unsplit attributes.
        // Initially all attributes are remained.
        for (int i = 0; i < this.trainData.entries.get(0).attributes.size(); i ++) {
            attributes.add(i);
        }

        this.start = new Node();
        this.root = ID3(this.trainData, attributes);
    }

    /**
     * Funtion to start testing the test dataset.
     * @return The accuracy.
     */
    public double startTesting() {
        this.confusionMatrix = new HashMap<>();
        double correct = 0;
        double all = 0;
        for (Entry e: this.testData.entries) {
            String predictedLabel = getPrediction(e, this.root);
            if (predictedLabel.equals(e.label)) {
                correct ++;
            } else {
                System.out.print("Miss classifying [ ");
                for (CellData d: e.attributes) {
                    System.out.print(d.value + ", ");
                }
                System.out.print(e.label + "]\tas [" + predictedLabel + "]\n");
            }
            Pair<String, String> pair = new Pair<>(e.label, predictedLabel);
            updateConfusionMatrix(pair);
            all ++;
        }
        double accuracy = correct / all;
        System.out.println("Accuracy: " + accuracy);
        return accuracy;
    }


    /**
     * For random forest testing purpose.
     * @param e Entry that random forest wants to get result on.
     * @return The predicted label of the input entry.
     */
    public String startTesting(Entry e) {
        String predictedLabel = getPrediction(e, this.root);
        return predictedLabel;
    }
}
