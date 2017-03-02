/**
 * Created by d_d on 3/1/17.
 */

/**
 * This class is a wrapper for categorical and continuous data.
 */
class CellData {
    // Undetermined type value.
    Object value;

    // Constructor for raw data and its type.
    CellData(String data, Boolean type)  {
        if (type) { // Categorical
            this.value = data;
        } else { // Continuous

            try {

                if (data.contains(",")) {
                    data = data.replace(",", ".");
                }

                this.value = Double.parseDouble(data);

            } catch (Exception e) {

                System.out.print(data + " can only be categorical feature!");
                this.value = data;
            }
        }
    }

    // Constructor for using utilities function inside the class.
    CellData() {
        this.value = null;
    }

    // Constructor for continuous data.
    CellData(Double value) {
        this.value = value;
    }

    // Constructor for categorical data.
    CellData(String value) {
        this.value = value;
    }

    // Compare criteria for continuous data.
    int compare(CellData cellData1, CellData cellData2) {
        return Double.compare((Double) cellData1.value, (Double) cellData2.value);
    }

    // Get mean of two continuous data.
    double getMean(CellData cellData1, CellData cellData2) {
        return (((Double) cellData1.value) + ((Double) cellData2.value)) / 2.0;
    }

}
