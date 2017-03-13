import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by d_d on 3/13/17.
 */
public class Main {

    public static void main(String[] args) throws Exception {
        ArrayList<Boolean> typeSpecification = new ArrayList<>(Arrays.asList(true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false));
        ArrayList<Boolean> chosenAttributes = new ArrayList<>(Arrays.asList(false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true));

        String header = "user;gender;age;how_tall_in_meters;weight;body_mass_index;x1;y1;z1;x2;y2;z2;x3;y3;z3;x4;y4;z4;class";
        RFMapReduce rfMapReduce = new RFMapReduce(typeSpecification, chosenAttributes, ";", header);

        rfMapReduce.setTrainSubsetFraction("0.67");

        rfMapReduce.RFDriver(args);

        rfMapReduce.accuracyCalculation(args[1] + "/part-r-00000");

    }
}
