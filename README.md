# Random Forest on MapReduce
A Random Forest MapReduce implementation.

Used DecisionTree in another repository of mine.

#Instructions
####command line parameters

`[input training data folder] [output folder] [path to test data] [number of trees]`

For example:
`input output /path/to/test.csv 5`

####Steps:
1. Specifying type for each attributes is required.
2. Specifying selected splitting attributes is required.
3. After creating the instance of a `RFMapReduce`, calling `setTrainSubsetFraction()` is required, usually "0.67".
4. Call `RFDriver()` to execute.
5. (Optional) Calculate accuracy.

#Structures
1. Read train data from a CSV file.                                                                          
2. Build n InputSplits for n trees, n is a command line argument.                                            
   1. Use customized InputFormat.getSplits() to create n InputSplits. So the framework would call n mappers.
   2. Use customized RecordReader.nextKeyValue() to create 2/3 subset of the training data with replacement.
   3. When Mapper.run() is calling nextKeyValue(), this method directly return 2/3 of the data.                
3. Each InputSplit would assign to a mappper.                                                                
4. After receiving data, each mapper start to build tree and produce prediction for test dataset.            
   (Each mapper is only going to receive one key/value pair from RecordReader.)                              
5. Pass the test data and label as key and value to reducer.                                                 
6. Reducer counts the majority label according to key.                                                       
7. Write results to output file.                                                                             
