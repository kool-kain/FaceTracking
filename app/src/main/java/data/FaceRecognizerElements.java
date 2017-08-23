package data;


import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;

public class FaceRecognizerElements {

    MatVector matVector;
    Mat labels;

    public FaceRecognizerElements(MatVector matVector, Mat labels) {
        this.matVector = matVector;
        this.labels = labels;
    }

    public MatVector getMatVector() {
        return matVector;
    }

    public void setMatVector(MatVector matVector) {
        this.matVector = matVector;
    }

    public Mat getLabels() {
        return labels;
    }

    public void setLabels(Mat labels) {
        this.labels = labels;
    }
}
