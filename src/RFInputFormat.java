import org.apache.hadoop.fs.BlockLocation;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by d_d on 3/12/17.
 *
 * This class is customized for processing training input data so MapReduce
 * framework would be assigning each input split to a certain number of mappers.
 *
 * This class is only for creating input split so the number of mappers can get
 * controlled.
 *
 * Actual data reading is done in RFRecordReader.
 *
 */
public class RFInputFormat extends FileInputFormat<IntWritable, Text> {
    /**
     * Overridden method for customized record reader.
     * @param split For each input split, a record reader is returned as responsible
     *              to process this input split.
     * @param context The job context.
     * @return RecordReader Indicate which reader to use.
     */
    @Override
    public RecordReader<IntWritable, Text> createRecordReader(InputSplit split, TaskAttemptContext context) {
        return (RecordReader) new RFRecordReader();
    }

    /**
     * Overridden method for creating n number of InputSplits the same as the number of
     * trees(Argument from command line.) This is essential for mapper as each input
     * split is assign to a mapper in MapReduce framework.
     * @param job The job context.
     * @return  A list of input split.
     * @throws IOException In case of IOException.
     */
    @Override
    public List<InputSplit> getSplits(JobContext job) throws IOException {
        List<FileStatus> files = listStatus(job);
        int numOfTrees = Integer.parseInt(job.getConfiguration().get("numOfTrees"));

        List<InputSplit> returnInputSplits = new ArrayList<InputSplit>();

        for (int i = 0; i < numOfTrees; i ++) {
            FileStatus file = files.get(0);
            Path path = file.getPath();
            FileSystem fs = path.getFileSystem(job.getConfiguration());

            // We don't actually need to use start/len for blkLocations. We're not reading
            // the training set according to locations. Reading rules is done by RecordReader.
            BlockLocation[] blkLocations = fs.getFileBlockLocations(file, 0, 1);

            returnInputSplits.add(new FileSplit(path, 0, 1, blkLocations[0].getHosts()));
        }
        return returnInputSplits;
    }
}
