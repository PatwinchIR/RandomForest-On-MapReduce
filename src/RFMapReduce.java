import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.*;

/**
 * Created by d_d on 3/11/17.
 *
 * This class is Map Reduce version of Random Forest.
 * RFMapReduce is an independent class with RandomForest(Normal version).
 *
 * Used DecisionTree class.
 *
 * This structure is as follows:
 * ===============================================================================================================
 * | 1. Read train data from a CSV file.
 * | 2. Build n InputSplits for n trees, n is a command line argument.
 * |    1). Use customized InputFormat.getSplits() to create n InputSplits. So the framework would call n mappers.
 * |    2). Use customized RecordReader.nextKeyValue() to create 2/3 subset of the training data with replacement.
 * |    3). When Mapper.run() is calling nextKeyValue(), this method directly return the 2/3 data.
 * | 3. Each InputSplit would assign to a mappper.
 * | 4. After receiving data, each mapper start to build tree and produce prediction for test dataset.
 * |    (Each mapper is only going to receive one key/value pair from RecordReader.)
 * | 5. Pass the test data and label as key and value to reducer.
 * | 6. Reducer counts the majority label according to key.
 * | 7. Write results to output file.
 * ===============================================================================================================
 * TODO: Support more results analysis, ex: Confusion Matrix, etc.
 */
public class RFMapReduce {

    // Configuration.
    public Configuration conf;

    //MapReduce Job.
    public Job job;

    // Attributes' type(categorical/continuous) specification.
    private String typeSpecification;

    // Data CSV file delimiter.
    private String delimiter;

    // The useful choosen attributes.
    private String chosenAttributes;

    // CSV datafile header.
    private String header;

    // The random factor for training subset selection.
    public String trainSubsetFraction;

    // Indicates the Random subspace in Random Forest.
    public int attrSubspaceNum;

    /**
     * Mapper class for random forest.
     * Each RFMapper Instance is for one decision tree.
     */
    public static class RFMapper extends Mapper<IntWritable, Text, Text, Text> {
        // Configuration.
        Configuration conf;

        // Attributes' type(categorical/continuous) specification.
        ArrayList<Boolean> typeSpec;

        // The useful choosen attributes.
        ArrayList<Boolean> chosenAttrs;

        // Indicates the Random subspace in Random Forest.
        int attrSubspaceNum;

        // Store attributes' name if the data has a header.
        public ArrayList<String> attributesName;

        // Data CSV file delimiter.
        String delimiter;

        // Training data.
        Entries train;

        // DecisionTree instance for this mapper.
        DecisionTree dt;

        /**
         * Overridden setup method to setup and initialize decision tree.
         * @param context The job context.
         * @throws IOException In case of IOException.
         */
        @Override
        protected void setup(Context context) throws IOException {
            conf = context.getConfiguration();
            typeSpec = new ArrayList<>();
            chosenAttrs = new ArrayList<>();

            // Build type specification from configuration 0/1 sequence.
            String tempTypeSpec = conf.get("typeSpecification");
            for (int i = 0; i < tempTypeSpec.length(); i ++) {
                typeSpec.add(tempTypeSpec.charAt(i) == '1');
            }

            // Build chosen attributes from configuration 0/1 sequence.
            String tempChosenAttrs = conf.get("chosenAttributes");
            for (int i = 0; i < tempChosenAttrs.length(); i ++) {
                chosenAttrs.add(tempChosenAttrs.charAt(i) == '1');
            }

            delimiter = conf.get("delimiter");
            attrSubspaceNum = Integer.parseInt(conf.get("attrSubspaceNum"));

            // Process header if there is, from Configuration.
            String header = conf.get("header");
            if (!header.equals("null")) {
                attributesName = new ArrayList<>();
                attributesName.addAll(Arrays.asList(header.split(delimiter)));
            } else {
                attributesName = null;
            }

            train = new Entries();

            // Initialization.
            dt = new DecisionTree(typeSpec, chosenAttrs, delimiter, true);
            dt.attrSubspaceNum = attrSubspaceNum;

            // Load testing data directly into DecisionTree Instance.
            URI[] localFiles = context.getCacheFiles();
            dt.loadData(false, localFiles[0].getPath(), false);
        }

        /**
         * After being called, using prepared data to grow the tree.
         * @param key   Just to satisfy the framework, no actual use.
         * @param value The training data generated from nextKeyValue() in RFRecordReader.
         * @param context   The job context.
         * @throws IOException In case of IOException.
         * @throws InterruptedException In case of InterruptedException.
         */
        @Override
        public void map(IntWritable key, Text value, Context context) throws IOException, InterruptedException {
            // The value comes from nextKeyValue is the whole training dataset which is a big Text stream seperated
            // by line breaker("\n") as each entry.
            String[] rawLines = (value.toString()).split("\n");
            List<String[]> rawEntries = new ArrayList<>();

            for (String s: rawLines) {
                rawEntries.add(s.split(delimiter));
            }

            for (int j = 0; j < rawEntries.size(); j ++) {
                String[] s = rawEntries.get(j);
                int i;

                // Just being lazy, should deal with this edge case in nextKeyValue() in RFRecordReader.
                if (s.length != (typeSpec.size() + 1)) {
                    continue;
                }

                Entry newEntry = new Entry();
                for (i = 0; i < s.length - 1; i ++) {
                    newEntry.attributes.add(new CellData(s[i], typeSpec.get(i)));
                }

                newEntry.label = s[i];
                train.entries.add(newEntry);
            }

            dt.trainData = train;
            dt.attributesName = attributesName;

            dt.startTraining();

            // Uncomment this line to print tree structure.
            //dt.preorderTraversePrint(dt.start, dt.root, -1, false, true);

            for (Entry e: dt.testData.entries) {
                context.write(new Text(e.toString(delimiter)), new Text(dt.startTesting(e)));
            }
        }

    }

    /**
     * Reducer class for random forest.
     * Each reducer is for one entry of test data.
     */
    public static class RFReducer extends Reducer<Text, Text, Text, Text> {
        // Configuration.
        Configuration conf;

        // Predicted labels hash map to count the majority label.
        Map<Text, Integer> predictedLabels;

        /**
         * A utility that facilitates the counting process in a hash map for certain key.
         * Same as in python Collections.Counter().
         * @param hashMap The hash map that needs to be updated.
         * @param key The key that needs to be updated.
         * @return The updated hash map.
         */
        private Map<Text, Integer> Counter(Map<Text, Integer> hashMap, Text key) {
            Map<Text, Integer> temp = new HashMap<>(hashMap);
            if (temp.containsKey(key)) {
                int count = temp.get(key);
                temp.put(key, ++ count);
            } else {
                temp.put(key, 1);
            }
            return temp;
        }

        /**
         * Overridden method to initialize predictedLabels.
         * @param context The job context.
         */
        @Override
        protected void setup(Context context) {
            conf = context.getConfiguration();
            predictedLabels = new HashMap<>();
        }

        /**
         * Overridden method to collect all labels from the n trees(Mappers)
         * and write the majority one to the output file.
         * @param key   The test data entry.
         * @param values    The predicted labels from n trees(Mappers).
         * @param context   The job context.
         * @throws IOException In case of IOException.
         * @throws InterruptedException In case of InterruptedException.
         */
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // Updating the majority labels from values.
            for (Text val: values) {
                predictedLabels = Counter(predictedLabels, val);
            }

            // Get the final majority label.
            Text finalLabel = Collections.max(predictedLabels.entrySet(), Map.Entry.comparingByValue()).getKey();

            context.write(key, finalLabel);
        }
    }

    /**
     * Used to set random forest training data set fraction.
     * @param trainSubsetFraction The string for the fraction.
     */
    public void setTrainSubsetFraction(String trainSubsetFraction) {
        this.trainSubsetFraction = trainSubsetFraction;
    }

    /**
     * The driver method used to start MapReduce job.
     * @param args  The command line arguments.
     * @return Indicate if the job is completely successfully.
     * @throws Exception In case of Exception.
     */
    public int RFDriver(String[] args) throws Exception {
        this.conf = new Configuration();

        // Configuration are used to pass in parameters for Mapper and Reducer.
        this.conf.set("numOfTrees", args[3]);
        this.conf.set("trainSubsetFraction", trainSubsetFraction);
        this.conf.set("delimiter", delimiter);
        this.conf.set("typeSpecification", typeSpecification);
        this.conf.set("chosenAttributes", chosenAttributes);
        this.conf.set("attrSubspaceNum", "" + attrSubspaceNum);
        this.conf.set("header", header);

        this.job = Job.getInstance(conf, "RandomForest");
        this.job.setJarByClass( RFMapReduce.class);
        this.job.setMapperClass(RFMapper.class);
        this.job.setReducerClass(RFReducer.class);

        // Train data path.
        FileInputFormat.addInputPath(this.job, new Path(args[0]));

        // Set InputFormat class as customized one.
        this.job.setInputFormatClass(RFInputFormat.class);

        // Output path.
        FileOutputFormat.setOutputPath(this.job, new Path(args[1]));

        this.job.setMapOutputKeyClass(Text.class);
        this.job.setMapOutputValueClass(Text.class);

        this.job.setOutputKeyClass(Text.class);
        this.job.setOutputValueClass(Text.class);

        // Add testing file as cached file for Mapper to access.
        this.job.addCacheFile(new URI(args[2]));

        int returnValue = this.job.waitForCompletion(true) ? 0 : 1;

        if (this.job.isSuccessful()) {
            System.out.println("Job was successful");
        } else {
            System.out.println("Job was not successful");
        }

        return returnValue;
    }

    /**
     * Read tested result from output file to calculate accuracy.
     * @param filePath The output file path. Usually as default.
     * @throws IOException In case of IOException.
     */
    public void accuracyCalculation(String filePath) throws IOException {
        BufferedReader fileReader = new BufferedReader(new FileReader(filePath));
        String line;
        double all = 0;
        double correct = 0;
        while ((line = fileReader.readLine()) != null) {
            all += 1;

            String[] tokens = line.split("\t");

            if (tokens.length != 2) {
                continue;
            }

            String[] temp = tokens[0].split(this.delimiter);
            String trueLabel = temp[temp.length - 1];
            String predictedLabel = tokens[1];

            if (trueLabel.equals(predictedLabel)) {
                correct += 1;
            }
        }
        System.out.println("Accuracy: " + correct / all);
        return;
    }

    /**
     * Constructor for RFMapReduce, must indicate header as a string delimited by the delimiter appeared in the data.
     * @param typeSpecification Attributes' type(categorical/continuous) specification.
     * @param chosenAttributes A boolean array indicates the attributes that user choose to use/ignore.
     * @param delimiter Data CSV file delimiter.
     * @param header The data header.(Attributes' name).
     * @throws Exception In case of Exception.
     */
    public RFMapReduce(ArrayList<Boolean> typeSpecification, ArrayList<Boolean> chosenAttributes, String delimiter, String header) throws Exception {
        this.typeSpecification = "";
        this.chosenAttributes = "";
        for (int i = 0; i < typeSpecification.size(); i ++) {
            this.typeSpecification += (typeSpecification.get(i) ? "1" : "0");
            this.chosenAttributes += (chosenAttributes.get(i) ? "1" : "0");
        }

        this.attrSubspaceNum = (int) Math.sqrt(this.typeSpecification.length());

        this.delimiter = delimiter;

        this.header = header;
    }

    /**
     * Constructor for RFMapReduce, without header information.
     * @param typeSpecification Attributes' type(categorical/continuous) specification.
     * @param chosenAttributes A boolean array indicates the attributes that user choose to use/ignore.
     * @param delimiter Data CSV file delimiter.
     * @throws Exception In case of Exception.
     */
    public RFMapReduce(ArrayList<Boolean> typeSpecification, ArrayList<Boolean> chosenAttributes, String delimiter) throws Exception {
        this.typeSpecification = "";
        this.chosenAttributes = "";
        for (int i = 0; i < typeSpecification.size(); i ++) {
            this.typeSpecification += (typeSpecification.get(i) ? "1" : "0");
            this.chosenAttributes += (chosenAttributes.get(i) ? "1" : "0");
        }

        this.attrSubspaceNum = (int) Math.sqrt(this.typeSpecification.length());

        this.delimiter = delimiter;

        this.header = null;
    }
}
