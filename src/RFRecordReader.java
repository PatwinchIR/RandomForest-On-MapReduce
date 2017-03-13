import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.util.LineReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by d_d on 3/12/17.
 *
 * This class is essential for creating subset with replacement
 * of training data for each tree(in this case, mapper).
 */
public class RFRecordReader extends RecordReader<IntWritable, Text> {
    // Only assign value once for mapper who's calling RFRecordReader.
    private boolean returned;

    // For line reader's input. Can be set to max.
    private int maxLineLength;

    // The random factor for training subset selection.
    private double trainSubsetFraction;

    // Key is not really used.
    private IntWritable key = new IntWritable();

    // Use value to store subset of data.(Generated with replacement.)
    private Text value = new Text();

    private LineReader in;

    @Override
    public IntWritable getCurrentKey() {
        return this.key;
    }

    @Override
    public Text getCurrentValue() {
        return this.value;
    }

    @Override
    public boolean nextKeyValue() throws IOException {
        if (returned) {
            return false;
        } else {
            List<String> entries = new ArrayList<>();

            List<String> bagging = new ArrayList<>();

            Text line = new Text();
            int size = 1;
            // Read all data.
            while (size != 0) {
                size = in.readLine(line, maxLineLength);
                entries.add(line.toString());
            }


            // Below is generating a fraction(subset) of training data.
            int trainSubsetSize = (int) (entries.size() * trainSubsetFraction);
            ArrayList<Integer> trainIndexes = new ArrayList<>();

            for (int i = 0; i < trainSubsetSize; i ++) {
                Integer index = (int) (Math.random() * (entries.size()));

                while (trainIndexes.contains(index)) {
                    index = (int) (Math.random() * (entries.size()));
                }

                trainIndexes.add(index);
                bagging.add(entries.get(index));
            }

            // Generating the passable string, to give it back to mapper.
            String writableEntries = "";
            for (String s: bagging) {
                writableEntries += (s + "\n");
            }

            value = new Text(writableEntries);

            returned = true;
            return true;
        }
    }

    @Override
    public float getProgress() throws IOException {
        return 0;
    }

    /**
     * Initialization
     * @param genericSplit  The input split that's assigned to RFRecordReader.
     * @param context The job context.
     * @throws IOException In case of IOException.
     */
    @Override
    public void initialize(InputSplit genericSplit, TaskAttemptContext context) throws IOException {
        FileSplit split = (FileSplit) genericSplit;
        Configuration conf = context.getConfiguration();

        this.trainSubsetFraction = Double.parseDouble(conf.get("trainSubsetFraction"));

        this.maxLineLength = conf.getInt("mapred.linerecordreader.maxlength", Integer.MAX_VALUE);

        final Path file = split.getPath();
        FileSystem fs = file.getFileSystem(conf);
        FSDataInputStream fileIn = fs.open(split.getPath());

        in = new LineReader(fileIn, conf);

        returned = false;
    }

    @Override
    public void close() throws IOException {

    }

}
