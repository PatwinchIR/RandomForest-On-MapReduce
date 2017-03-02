/**
 * Created by d_d on 3/1/17.
 */

import java.util.ArrayList;
import java.util.List;

/**
 * This class is specifically for row in dataset, attributes is a list of Double value, because of
 * continuous/numerical attributes.
 * label is a String indicating the label.
 */
class Entry {
    List<CellData> attributes;
    String label;

    Entry() {
        this.attributes = new ArrayList<>();
        this.label = null;
    }
}
