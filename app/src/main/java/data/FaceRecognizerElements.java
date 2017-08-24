package data;


import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;

import java.io.Serializable;

public class FaceRecognizerElements implements Serializable {
    private static final long serialVersionUID = 1L;
    MatVector matVector;
    Mat labels;

    public FaceRecognizerElements() {
        matVector = new MatVector();
        labels = new Mat();
    }

    public FaceRecognizerElements(FaceRecognizerElements faceRecognizerElements) {
        this.matVector = faceRecognizerElements.getMatVector();
        this.labels = faceRecognizerElements.getLabels();
    }

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
