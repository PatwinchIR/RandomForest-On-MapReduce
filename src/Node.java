/**
 * Created by d_d on 3/1/17.
 */

import java.util.*;

/**
 * This class is for Node of decision tree.
 * It stores information about current entropy, current examples labels' kinds and numbers.
 * it also stores next best attribute to split, and the splitting boundary.
 */
class Node {
    // Attributes' type(categorical/continuous) specification.
    List<Boolean> typeSpecification;

    // Entropy calculated on labelsCount which is formed by splitting boundary.
    double entropy;

    // Using the splitting boundary form a labelsCount.
    // It's a hash map, the key is label, the value is the number of it.
    Map<String, Integer> labelsCount;

    // Leaf node's label, which is used to produce prediction. NULL if non-leaf nodes.
    String label;

    // To tag if current node needs more splitting.
    boolean isConsistent;

    // The best attribute that needs to be split.
    int bestAttribute;

    // The decision boundary for the best attribute. Also using this to binarize the data.
    CellData decision;

    // Left child.
    Node left;

    // Right child.
    Node right;

    /**
     * Utility of LOG function, support any base.
     * @param x Log parameter.
     * @param base Base, should be integer.
     * @return The log result.
     */
    private static double log(double x, int base) {
        return (Math.log(x) / Math.log(base));
    }

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
     * For current examples, generate the label count, for later calculating entropy and consistency check.
     * @param examples The examples that current node received.
     */
    private void processLabels(Entries examples) {
        List<String> labels = new ArrayList<>();
        this.labelsCount = new HashMap<>();
        for (Entry e: examples.entries) {
            this.labelsCount = Counter(this.labelsCount, e.label);
            labels.add(e.label);
        }
        if (this.labelsCount.size() == 1) {

            // If only one label exists in current example then set the prediction label to it.
            this.label = labels.get(0);

            // No need to split more, current node is consistent with examples.
            this.isConsistent = true;

        } else {
            // Received more than 1 labels, meaning not consistent, set the tag to false.
            this.isConsistent = false;
        }
    }

    /**
     * Calculate entropy for a labelsCount.
     * @param n The total number of labels.
     * @param labelsCount The hash map of labels, the key is label, the value is the number of it.
     * @return The labelsCount's entropy.
     */
    private double calculateEntropy(Integer n, Map<String, Integer> labelsCount) {
        double entropy = 0;
        Iterator it = labelsCount.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry pair = (Map.Entry)it.next();
            Double count = ((Integer) pair.getValue()) * 1.0; // * 1.0 is to convert it to double type.
            double p = count / n;
            entropy -= (p * log(p, 2));
        }
        return entropy;
    }

    /**
     * The main function to find the next best splitting attribute.
     * I put it into Node because every node would receive a set of examples when it's created and wouldn't be
     * changed later.
     * @param examples The examples that current node received after its parent's splitting.
     * @param attributes The remaining attributes that haven't been spitted before.
     */
    private void findBestSplitAttr(Entries examples, ArrayList<Integer> attributes) {

        // minEntropy over all attributes and all candidate boundaries.
        double minEntropy = Double.MAX_VALUE;

        // Traverse all remaining attributes.
        for (Integer attrIdx: attributes) {

            if (!this.typeSpecification.get(attrIdx)) {     // Continuous
                // Sort examples according to current attributes.
                Collections.sort(examples.entries, new Comparator<Entry>() {
                    @Override
                    public int compare(Entry o1, Entry o2) {
                        return new CellData().compare(o1.attributes.get(attrIdx),
                                o2.attributes.get(attrIdx));
                    }
                });

                // Trying all candidate boundaries.
                for (int i = 1; i < examples.entries.size(); i++) {

                    // Discretise examples into binary.
                    Map<String, Integer> pos = new HashMap<>();
                    Map<String, Integer> neg = new HashMap<>();

                    for (int j = 0; j < examples.entries.size(); j++) {
                        String newLabel = examples.entries.get(j).label;
                        if (j < i) {
                            pos = Counter(pos, newLabel);
                        } else {
                            neg = Counter(neg, newLabel);
                        }
                    }

                    // Calculate pos and neg entropy.
                    double posFraction = (i * 1.0) / examples.entries.size();
                    double posEntropy = posFraction * calculateEntropy(i, pos);
                    double negEntropy = (1 - posFraction) * calculateEntropy(examples.entries.size() - i, neg);

                    // Updating the minEntropy.
                    if ((posEntropy + negEntropy) < minEntropy) {
                        minEntropy = posEntropy + negEntropy;
                        this.decision = new CellData(new CellData().getMean(examples.entries.get(i - 1).attributes.get(attrIdx), examples.entries.get(i).attributes.get(attrIdx)));
                        this.bestAttribute = attrIdx;
                    }
                }
            } else {        // Categorical
                Set<String> categories = new HashSet<>();

                // Get all categories.
                for (int i = 0; i < examples.entries.size(); i ++) {
                    String data = (String) examples.entries.get(i).attributes.get(attrIdx).value;
                    categories.add(data);
                }

                // Find the best category to split.
                for (String category: categories) {

                    Map<String, Integer> pos = new HashMap<>();
                    Map<String, Integer> neg = new HashMap<>();

                    for (int i = 0; i < examples.entries.size(); i ++) {
                        String newLabel = examples.entries.get(i).label;
                        String data = (String) examples.entries.get(i).attributes.get(attrIdx).value;
                        if (category.equals(data)) {
                            pos = Counter(pos, newLabel);
                        } else {
                            neg = Counter(pos, newLabel);
                        }
                    }

                    // Calculate pos and neg entropy.
                    double posFraction = (pos.size() * 1.0) / examples.entries.size();
                    double posEntropy = posFraction * calculateEntropy(pos.size(), pos);
                    double negEntropy = (1 - posFraction) * calculateEntropy(neg.size(), neg);

                    // Updating the minEntropy.
                    if ((posEntropy + negEntropy) < minEntropy) {
                        minEntropy = posEntropy + negEntropy;
                        this.decision = new CellData(category);
                        this.bestAttribute = attrIdx;
                    }
                }
            }
        }
    }

    /**
     * Constructor for Node when it needs to receive examples and remaining attributes.
     * @param examples The remaining examples after its parent's splitting.
     * @param attributes The remaining attributes after its parent's splitting.
     */
    Node(Entries examples, ArrayList<Integer> attributes, ArrayList<Boolean> typeSpecification) {
        this.left = null;
        this.right = null;
        this.label = null;
        this.typeSpecification = typeSpecification;

        processLabels(examples);

        this.entropy = calculateEntropy(examples.entries.size(), this.labelsCount);

        findBestSplitAttr(examples, attributes);
    }

    /**
     * Constructor of Node when there's no examples left.
     */
    Node() {
        this.left = null;
        this.right = null;
        this.label = null;
        this.typeSpecification = new ArrayList<>();
        this.entropy = 0;
        this.labelsCount = new HashMap<>();
        this.isConsistent = true;
        this.bestAttribute = 0;
        this.label = "";
    }
}
