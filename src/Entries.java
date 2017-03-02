/**
 * Created by d_d on 3/1/17.
 */

import java.util.ArrayList;
import java.util.List;

/**
 * This class is specifically for dataset, each entry is a row in dataset, seperated as attributes and
 * correspongding label(also called target attribute in some ID3 algorithm tutorials).
 */
class Entries {
    List<Entry> entries;

    Entries() {
        this.entries = new ArrayList<>();
    }
}
