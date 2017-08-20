package resources_manager;

import android.content.res.Resources;
import android.support.annotation.Nullable;
import android.util.Log;

import com.tmf.dml.facetracking.R;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

public class ImagesProvider {
    private static final String TAG = "ImagesProvider::class";
    private Resources resources;
    private R.drawable drawableResources;
    private ArrayList<File> trainingMaleArray;
    private ArrayList<File> trainingFemaleArray;

    public ImagesProvider(Resources resources, File trainingDir) {

        this.resources = resources;

        HashMap<Integer, String> hMapMaleImages = new HashMap<Integer, String>();
        HashMap<Integer, String> hMapFemaleImages = new HashMap<Integer, String>();
        trainingMaleArray = new ArrayList<File>();
        trainingFemaleArray = new ArrayList<File>();

        Field[] drawables = R.drawable.class.getFields();
        R.drawable drawableResources = new R.drawable();
        for (Field f : drawables) {
            try {
                if (f.getName().startsWith("subject11")) {
                    hMapFemaleImages.put(f.getInt(drawableResources), f.getName());
                } else if (f.getName().startsWith("subject")) {
                    hMapMaleImages.put(f.getInt(drawableResources), f.getName());
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        trainingFemaleArray = generateTrainingDirWithImages(hMapFemaleImages, trainingDir);
        trainingMaleArray = generateTrainingDirWithImages(hMapMaleImages, trainingDir);

        Log.d(TAG, "Male size: " + trainingMaleArray.size() +
                "Female size: " + trainingFemaleArray.size());
    }

    @Nullable
    private ArrayList<File> generateTrainingDirWithImages(HashMap<Integer, String> hMap, File trainingDir) {

        ArrayList<File> resultTraining = new ArrayList<File>();
        try {
            for (Entry<Integer, String> entry : hMap.entrySet()) {
                InputStream is = resources.openRawResource(entry.getKey());
                File imageFile = new File(trainingDir, entry.getValue());
                FileOutputStream os = new FileOutputStream(imageFile);

                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }
                is.close();
                os.close();

                resultTraining.add(imageFile);
            }

        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Unable to load training images. Exception thrown: " + e);
            return null;
        }

        return resultTraining;
    }

    public MatVector getMaleImages() {
        MatVector malePics = new MatVector(trainingMaleArray.size());

        Mat labels = new Mat(malePics.size(), 1, CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();

        int counter = 0;

        for (File image : trainingMaleArray) {

            Mat imgFromResource = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);

            //Named as subjectXX, so we label as XX
            int label = Integer.parseInt(image.getName().substring(7, 8));
            Log.d(TAG, "Path: " + image.getAbsolutePath() + " || Label: " + label);
            malePics.put(counter, imgFromResource);

            labelsBuf.put(counter, label);

            counter++;
        }
        return malePics;
    }

    public MatVector getFemaleImages() {
        MatVector femalePics = new MatVector(trainingFemaleArray.size());

        Mat labels = new Mat(femalePics.size(), 1, CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();

        int counter = 0;

        for (File image : trainingFemaleArray) {
            Mat imgFromResource = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);

            //Named as subjectXX, so we label as XX
            int label = Integer.parseInt(image.getName().substring(7, 8));

            femalePics.put(counter, imgFromResource);

            labelsBuf.put(counter, label);

            counter++;
        }
        return femalePics;
    }
}
