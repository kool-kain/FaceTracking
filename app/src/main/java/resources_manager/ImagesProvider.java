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
import java.io.Serializable;
import java.lang.reflect.Field;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import data.FaceRecognizerElements;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

public class ImagesProvider implements Serializable {
    private static final String TAG = "ImagesProvider::class";
    public static final Integer LABEL_MALE = 0;
    public static final Integer LABEL_FEMALE = 1;
    private ArrayList<File> trainingMaleArray;
    private ArrayList<File> trainingFemaleArray;

    public enum Emotions {
        SAD("sad", 0),
        HAPPY("happy", 1),
        NORMAL("normal", 2),
        SLEEPY("sleepy", 3),
        SURPRISED("surprised", 4);

        private final String emotion;
        private final Integer tag;

        Emotions(String emotion, Integer tag) {
            this.emotion = emotion;
            this.tag = tag;
        }

        public String getEmotion() {
            return this.emotion;
        }

        public Integer getTag() {
            return this.tag;
        }
    }

    public void generateFiles(Resources resources, File trainingDir) {
        HashMap<Integer, String> hMapMaleImages = new HashMap<Integer, String>();
        HashMap<Integer, String> hMapFemaleImages = new HashMap<Integer, String>();
        trainingMaleArray = new ArrayList<File>();
        trainingFemaleArray = new ArrayList<File>();

        Field[] drawables = R.drawable.class.getFields();
        R.drawable drawableResources = new R.drawable();
        for (Field f : drawables) {
            try {
                if (f.getName().startsWith("male")) {
                    hMapMaleImages.put(f.getInt(drawableResources), f.getName());
                } else if (f.getName().startsWith("female")) {
                    hMapFemaleImages.put(f.getInt(drawableResources), f.getName());
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        trainingFemaleArray = generateTrainingDirWithImages(resources, hMapFemaleImages, trainingDir);
        trainingMaleArray = generateTrainingDirWithImages(resources, hMapMaleImages, trainingDir);

        Log.d(TAG, "Male size: " + trainingMaleArray.size() +
                "Female size: " + trainingFemaleArray.size());
    }

    @Nullable
    private ArrayList<File> generateTrainingDirWithImages(Resources resources,
                                                          HashMap<Integer, String> hMap, File trainingDir) {

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

    @Nullable
    public FaceRecognizerElements getAllImagesGenderLabelled() {
        Integer maleSize = trainingMaleArray.size();
        Integer femaleSize = trainingFemaleArray.size();

        if (maleSize + femaleSize <= 0)
            return null;

        MatVector allPics = new MatVector(maleSize + femaleSize);
        Mat labels = new Mat(maleSize + femaleSize, 1, CV_32SC1);
        Mat imgFromResource = new Mat();
        IntBuffer labelsBuf = labels.createBuffer();

        int counter = 0;

        for (File image : trainingMaleArray) {

            imgFromResource = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            Log.d(TAG, "Path: " + image.getAbsolutePath() + " || Label: " + LABEL_MALE);
            allPics.put(counter, imgFromResource);

            labelsBuf.put(counter, LABEL_MALE);

            counter++;
        }

        for (File image : trainingFemaleArray) {

            imgFromResource = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            Log.d(TAG, "Path: " + image.getAbsolutePath() + " || Label: " + LABEL_FEMALE);
            allPics.put(counter, imgFromResource);

            labelsBuf.put(counter, LABEL_FEMALE);

            counter++;
        }
        imgFromResource.release();
        return new FaceRecognizerElements(allPics, labels);
    }

    @Nullable
    public FaceRecognizerElements getAllImagesEmotionsLabelled() {
        Integer maleSize = trainingMaleArray.size();
        Integer femaleSize = trainingFemaleArray.size();

        if (maleSize + femaleSize <= 0)
            return null;

        MatVector emotionsPics = new MatVector(maleSize + femaleSize);
        Mat labels = new Mat(maleSize + femaleSize, 1, CV_32SC1);
        Mat imgFromResource = new Mat();
        IntBuffer labelsBuf = labels.createBuffer();

        int counter = 0;
        int label = -1;

        for (File image : trainingMaleArray) {

            imgFromResource = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            label = determineLabel(image.getName());
            Log.d(TAG, "Path: " + image.getAbsolutePath() + " || Label: " + label);
            emotionsPics.put(counter, imgFromResource);

            labelsBuf.put(counter, label);

            counter++;
        }

        for (File image : trainingFemaleArray) {

            imgFromResource = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            label = determineLabel(image.getName());
            Log.d(TAG, "Path: " + image.getAbsolutePath() + " || Label: " + LABEL_FEMALE);
            emotionsPics.put(counter, imgFromResource);

            labelsBuf.put(counter, label);

            counter++;
        }

        imgFromResource.release();
        return new FaceRecognizerElements(emotionsPics, labels);
    }

    private Integer determineLabel(String fileName) {
        Integer label = fileName.contains(Emotions.SAD.getEmotion()) ? Emotions.SAD.getTag()
                : fileName.contains(Emotions.HAPPY.getEmotion()) ? Emotions.HAPPY.getTag()
                : fileName.contains(Emotions.NORMAL.getEmotion()) ? Emotions.NORMAL.getTag()
                : fileName.contains(Emotions.SLEEPY.getEmotion()) ? Emotions.SLEEPY.getTag()
                : fileName.contains(Emotions.SURPRISED.getEmotion()) ? Emotions.SURPRISED.getTag()
                //other options are considered normal
                : Emotions.NORMAL.getTag();
        return label;
    }
}
