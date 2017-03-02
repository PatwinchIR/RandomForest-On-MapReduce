/**
 * Created by d_d on 3/1/17.
 */
public class CellDataUtils {
    public int compare(CellData cellData1, CellData cellData2) {
        return Double.compare((Double) cellData1.value, (Double) cellData2.value);
    }

    public double getMean(CellData cellData1, CellData cellData2) {
        return (((Double) cellData1.value) + ((Double) cellData2.value)) / 2.0;
    }
}
