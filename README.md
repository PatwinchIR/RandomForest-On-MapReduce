# Random Forest on MapReduce
A Random Forest MapReduce implementation.

Used DecisionTree in another repository of mine.

__[NOTE]__: Random Forest wasn't supposed to fit into MapReduce framework and design logic, this is just a for-fun project and is not optimized. It works but the actually performance for this library might be really really bad.

# Instructions
#### command line parameters

`[input training data folder] [output folder] [path to test data] [number of trees]`

For example:
`input output /path/to/test.csv 5`

#### Steps:
1. Specifying type for each attributes is required.
2. Specifying selected splitting attributes is required.
3. After creating the instance of a `RFMapReduce`, calling `setTrainSubsetFraction()` is required, usually "0.67".
4. Call `RFDriver()` to execute.
5. (Optional) Calculate accuracy.

# Structures
1. Read train data from a CSV file.                                                                          
2. Build n InputSplits for n trees, n is a command line argument.                                            
   1. Use customized `InputFormat.getSplits()` to create n `InputSplit`s. So the framework would call n mappers.
   2. Use customized `RecordReader.nextKeyValue()` to create 2/3 subset of the training data with replacement.
   3. When `Mapper.run()` is calling `nextKeyValue()`, this method directly return 2/3 of the data.                
3. Each `InputSplit` would assign to a mappper.                                                                
4. After receiving data, each mapper start to build tree and produce prediction for test dataset.            
   (Each mapper is only going to receive one key/value pair from `RecordReader`.)                              
5. Pass the test data and label as key and value to `Reducer`.                                                 
6. `Reducer` counts the majority label according to key.                                                       
7. Write results to output file.                                                                             

# Notes
1. Use `process.py` to process the `smallerData.csv` file to get 80/20 train/test data(approximately label balanced).
2. Use all the jars in the `JARS` folder as this project's dependencies. (It's all hadoop 2.7.3 framework.)
