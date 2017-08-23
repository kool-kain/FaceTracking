package com.tmf.dml.facetracking;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.opencv_core;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import data.FaceRecognizerElements;
import resources_manager.ImagesProvider;

import static java.lang.Math.atan2;
import static java.lang.Math.sqrt;
import static org.bytedeco.javacpp.opencv_core.CV_PI;
import static org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.opencv.core.Core.FONT_HERSHEY_DUPLEX;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.imgproc.Imgproc.warpAffine;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String TAG = "FdActivity::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0);
    private static final Scalar TEXT_COLOR = new Scalar(255);
    private static final int JAVA_DETECTOR = 0;
    private static final float EYE_SX = 0.16f;
    private static final float EYE_SY = 0.26f;
    private static final float EYE_SW = 0.30f;
    private static final float EYE_SH = 0.28f;
    private static final double DESIRED_LEFT_EYE_Y = 0.14;
    private static final double DESIRED_LEFT_EYE_X = 0.19;
    private static final int FACE_WIDTH = 320;
    private static final int FACE_HEIGHT = 243;
    private static final double THERESOLD = 90;

    private MenuItem mItemFace50;
    private MenuItem mItemFace40;
    private MenuItem mItemType;
    private int detectorType = JAVA_DETECTOR;
    private String[] detectorName;
    private Mat matRgba;
    private Mat matGray;
    private Mat matDest;
    private Mat matCropFace;
    private MatOfRect faces;
    private Rect leftEyeRectangle;
    private Rect rightEyeRectangle;
    private FaceRecognizer faceRecognizerFace;
    private FaceRecognizer faceRecognizerEmotion;
    private ImagesProvider imagesProvider;
    private File fileCascadeFile;
    private CascadeClassifier cascadeClassifierFace, cascadeClassifierEye;
    private String msg;
    private IntPointer label;
    DoublePointer confidence;
    private float relativeFaceSize = 0.3f;
    private int absoluteFaceSize = 0;


    private CameraBridgeViewBase camOpenCvCameraView;

    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    cascadeClassifierFace = initClassifier(R.raw.haarcascade_frontalface_alt,
                            "haarcascade_frontalface_alt.xml");
                    cascadeClassifierEye = initClassifier(R.raw.haarcascade_eye,
                            "haarcascade_eye.xml");

                    imagesProvider = new ImagesProvider(getResources(),
                            getDir("trainingDir", Context.MODE_PRIVATE));

                    faceRecognizerFace = createFisherFaceRecognizer();
                    faceRecognizerFace.setThreshold(THERESOLD);
                    faceRecognizerEmotion = createFisherFaceRecognizer();
                    faceRecognizerEmotion.setThreshold(THERESOLD + 10);

                    initTrainers();

                    camOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Nullable
    private CascadeClassifier initClassifier(int id, String fileName) {
        CascadeClassifier cascadeClassifier;

        try {
            // load cascade file from application resources
            InputStream is = getResources().openRawResource(id);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            fileCascadeFile = new File(cascadeDir, fileName);
            FileOutputStream os = new FileOutputStream(fileCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            cascadeDir.delete();

        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Unable to loading classifier. Exception thrown: " + e);
            return null;
        }

        cascadeClassifier = new CascadeClassifier(fileCascadeFile.getAbsolutePath());
        if (!cascadeClassifier.load(fileCascadeFile.getAbsolutePath())) {
            Log.d(TAG, "Failed to load cascade classifier");
            cascadeClassifier = null;
        } else {
            Log.d(TAG, "Loaded cascade classifier from " + fileCascadeFile.getAbsolutePath());
        }

        return cascadeClassifier;
    }

    public FdActivity() {
        detectorName = new String[1];
        detectorName[JAVA_DETECTOR] = "Java";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.face_detect_surface_view);

        camOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        camOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        camOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (camOpenCvCameraView != null)
            camOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, baseLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        camOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        matGray = new Mat();
        matDest = new Mat();
        matRgba = new Mat();
        matCropFace = new Mat();

        rightEyeRectangle = new Rect(0, 0, 0);
        leftEyeRectangle = new Rect(0, 0, 0);

        label = new IntPointer(1);
        confidence = new DoublePointer(1);
    }

    @Override
    public void onCameraViewStopped() {
        matGray.release();
        matDest.release();
        matRgba.release();
        matCropFace.release();
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        matRgba = inputFrame.rgba();
        matGray = inputFrame.gray();
        matDest = matRgba.clone();

        Imgproc.equalizeHist(matGray, matGray);

        faces = new MatOfRect();

        if (absoluteFaceSize == 0) {
            int height = matGray.rows();
            if (Math.round(height * relativeFaceSize) > 0) {
                absoluteFaceSize = Math.round(height * relativeFaceSize);
            }
        }

        if (cascadeClassifierFace != null) {
            cascadeClassifierFace.detectMultiScale(matGray, faces, 1.1, 4, 0,
                    new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        } else {
            Log.d(TAG, "Missed classifier, therefore no face detection allowed");
        }

        Log.d(TAG, "Faces length: " + faces.toArray().length + ". AbsFaceSize: " + absoluteFaceSize);

        for (Rect rect : faces.toArray()) {
            if (detectEyes(rect)) {
                Mat matCropFaceGray = new Mat();
                Imgproc.cvtColor(matCropFace, matCropFaceGray, Imgproc.COLOR_BGRA2GRAY);
                Imgproc.resize(matCropFaceGray, matCropFace, new Size(FACE_WIDTH, FACE_HEIGHT));

                msg = "S: " + faceRecognizingGender() + ". Mood: " + faceRecognizingEmotion();

                drawFaceMarksAndText(rect, msg, 20);
                Log.d(TAG, "Detected: " + msg);
                matCropFaceGray.release();
            } else {
                Log.d(TAG, "Couldn't determine a completed face.");
            }
        }

        faces.release();
        matGray.release();
        matRgba.release();
        matCropFace.release();

        return matDest;
    }

    private void initTrainers() {
        //First we recognized gender by training with a vector of labelled faces
        FaceRecognizerElements faceRecognizerElements = imagesProvider.getAllImagesGenderLabelled();
        if (faceRecognizerElements != null && faceRecognizerElements.getMatVector() != null
                && faceRecognizerElements.getLabels() != null) {
            faceRecognizerFace.train(faceRecognizerElements.getMatVector(),
                    faceRecognizerElements.getLabels());
        } else {
            Log.d(TAG, "Couldn't training faces model");
        }

        //And we recognized emotions by training with a vector of labelled emotion
        FaceRecognizerElements faceRecognizerElementsEmotions = imagesProvider.getAllImagesEmotionsLabelled();
        if (faceRecognizerElementsEmotions != null && faceRecognizerElementsEmotions.getMatVector() != null
                && faceRecognizerElementsEmotions.getLabels() != null) {
            faceRecognizerEmotion.train(faceRecognizerElementsEmotions.getMatVector(),
                    faceRecognizerElementsEmotions.getLabels());
        } else {
            Log.d(TAG, "Couldn't training emotions model");
        }
    }

    private String faceRecognizingGender() {
        String recognizing = "Unknown";
        //Need to convert to Mat of JavaCV package in order to faceRecognizer to run
        opencv_core.Mat matJavaCv = new opencv_core.Mat((Pointer) null) {
            {
                address = matCropFace.getNativeObjAddr();
            }
        };

        if (matJavaCv != null) {

            label.put(-1);
            faceRecognizerFace.predict(matJavaCv, label, confidence);
            int predictedLabel = label.get();

            if (ImagesProvider.LABEL_MALE.equals(predictedLabel)) {
                recognizing = "Man";
            } else if (ImagesProvider.LABEL_FEMALE.equals(predictedLabel)) {
                recognizing = "Woman";
            }
        } else {
            Log.d(TAG, "Mat conversion failed. Couldn't predict face");
        }
        matJavaCv.close();
        matJavaCv.deallocate();
        return recognizing;
    }

    private String faceRecognizingEmotion() {
        String recognizing = "Unknown";
        //Need to convert to Mat of JavaCV package in order to faceRecognizer to run
        opencv_core.Mat matJavaCv = new opencv_core.Mat((Pointer) null) {
            {
                address = matCropFace.getNativeObjAddr();
            }
        };

        if (matJavaCv != null) {

            label.put(-1);
            faceRecognizerEmotion.predict(matJavaCv, label, confidence);
            int predictedLabel = label.get();

            if (ImagesProvider.Emotions.SAD.getTag().equals(predictedLabel))
                recognizing = ImagesProvider.Emotions.SAD.getEmotion();
            else if (ImagesProvider.Emotions.HAPPY.getTag().equals(predictedLabel))
                recognizing = ImagesProvider.Emotions.HAPPY.getEmotion();
            else if (ImagesProvider.Emotions.NORMAL.getTag().equals(predictedLabel))
                recognizing = ImagesProvider.Emotions.NORMAL.getEmotion();
            else if (ImagesProvider.Emotions.SLEEPY.getTag().equals(predictedLabel))
                recognizing = ImagesProvider.Emotions.SLEEPY.getEmotion();
            else if (ImagesProvider.Emotions.SURPRISED.getTag().equals(predictedLabel))
                recognizing = ImagesProvider.Emotions.SURPRISED.getEmotion();
        } else {
            Log.d(TAG, "Mat conversion failed. Couldn't predict emotion");
        }
        matJavaCv.close();
        matJavaCv.deallocate();
        return recognizing;
    }

    @NonNull
    private Boolean detectEyes(Rect rect) {
        matCropFace = matDest.submat(rect);
        MatOfRect leftEye = new MatOfRect(), rightEye = new MatOfRect();

        int leftX = Math.round(matCropFace.cols() * EYE_SX);
        int topY = Math.round(matCropFace.rows() * EYE_SY);
        int widthX = Math.round(matCropFace.cols() * EYE_SW);
        int heightY = Math.round(matCropFace.rows() * EYE_SH);
        int rightX = (int) Math.round(matCropFace.cols() * (1.0 - EYE_SX - EYE_SW));

        Mat topLeftOfFace = matCropFace.submat(new Rect(leftX, topY, widthX, heightY));
        Mat topRightOfFace = matCropFace.submat(new Rect(rightX, topY, widthX, heightY));

        if (cascadeClassifierEye != null) {
            cascadeClassifierEye.detectMultiScale(topLeftOfFace, leftEye);
            cascadeClassifierEye.detectMultiScale(topRightOfFace, rightEye);
        }
        Rect[] leftEyeArray = leftEye.toArray();
        if (leftEyeArray.length > 0) {
            leftEyeRectangle = leftEyeArray[0];
        } else {
            Log.d(TAG, "Couldn't find left eye. LeftEye length: " + leftEyeArray.length);
        }

        Rect[] rightEyeArray = rightEye.toArray();
        if (rightEyeArray.length > 0) {
            rightEyeRectangle = rightEyeArray[0];
        } else {
            Log.d(TAG, "Couldn't find right eye. RightEye length: " + rightEyeArray.length);
        }

        topLeftOfFace.release();
        topRightOfFace.release();
        return (leftEyeArray.length > 0) && (rightEyeArray.length > 0);
    }

    @Deprecated
    private void cropFace() {

        Point leftPoint = new Point(leftEyeRectangle.x + leftEyeRectangle.width / 2, leftEyeRectangle.y
                + leftEyeRectangle.height / 2);
        Point rightPoint = new Point(rightEyeRectangle.x + rightEyeRectangle.width / 2, rightEyeRectangle.y
                + rightEyeRectangle.height / 2);
        Point eyesCenter = new Point((leftPoint.x + rightPoint.x) * 0.5f,
                (leftPoint.y + rightPoint.y) * 0.5f);

        // Get the angle between the 2 eyes.
        double dy = (rightPoint.y - leftPoint.y);
        double dx = (rightPoint.x - leftPoint.x);
        double len = sqrt(dx * dx + dy * dy);
        double angle = atan2(dy, dx) * 180.0 / CV_PI;

        // Hand measurements shown that the left eye center should ideally be at roughly (0.19, 0.14) of a scaled face image.
        double desiredRightEyeX = (1.0f - DESIRED_LEFT_EYE_X);

        // Get the amount we need to scale the image to be the desired fixed size we want.
        double desiredLen = (desiredRightEyeX - DESIRED_LEFT_EYE_X) * FACE_WIDTH;
        double scale = desiredLen / len;

        // Get the transformation matrix for rotating and scaling the face to the desired angle & size.
        Mat rotatingMat = new Mat();
        rotatingMat = Imgproc.getRotationMatrix2D(eyesCenter, angle, scale);

        // Shift the center of the eyes to be the desired center between the eyes.
        double buff[] = new double[(int) (rotatingMat.total() * rotatingMat.channels())];
        rotatingMat.get(0, 2, buff);
        for (Double pixel : buff) {
            pixel = FACE_WIDTH * 0.5f - eyesCenter.x;
        }
        rotatingMat.put(0, 2, buff);

        buff = new double[(int) (rotatingMat.total() * rotatingMat.channels())];
        rotatingMat.get(1, 2, buff);
        for (Double pixel : buff) {
            pixel = FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenter.y;
        }
        rotatingMat.put(1, 2, buff);

        matCropFace = new Mat(FACE_WIDTH, FACE_HEIGHT, CV_8U, new Scalar(128));

        warpAffine(matDest, matCropFace, rotatingMat, matCropFace.size());

        rotatingMat.release();
    }

    private void drawFaceMarksAndText(Rect rect, String msg, final int LINE_WIDTH) {

        Imgproc.line(matDest, new Point(rect.x, rect.y), new Point(rect.x, rect.y + LINE_WIDTH), FACE_RECT_COLOR, 3);
        Imgproc.line(matDest, new Point(rect.x, rect.y), new Point(rect.x + LINE_WIDTH, rect.y), FACE_RECT_COLOR, 3);

        Imgproc.line(matDest, new Point(rect.x + rect.width, rect.y),
                new Point(rect.x + rect.width, rect.y + LINE_WIDTH), FACE_RECT_COLOR, 3);
        Imgproc.line(matDest, new Point(rect.x + rect.width, rect.y),
                new Point(rect.x + rect.width - LINE_WIDTH, rect.y), FACE_RECT_COLOR, 3);

        Imgproc.line(matDest, new Point(rect.x, rect.y + rect.height),
                new Point(rect.x, rect.y + rect.height - LINE_WIDTH), FACE_RECT_COLOR, 3);
        Imgproc.line(matDest, new Point(rect.x, rect.y + rect.height),
                new Point(rect.x + LINE_WIDTH, rect.y + rect.height), FACE_RECT_COLOR, 3);

        Imgproc.line(matDest, new Point(rect.x + rect.width, rect.y + rect.height),
                new Point(rect.x + rect.width, rect.y + rect.height - LINE_WIDTH), FACE_RECT_COLOR, 3);
        Imgproc.line(matDest, new Point(rect.x + rect.width, rect.y + rect.height),
                new Point(rect.x + rect.width - LINE_WIDTH, rect.y + rect.height), FACE_RECT_COLOR, 3);

        int font = FONT_HERSHEY_DUPLEX;
        Size s = Imgproc.getTextSize(msg, font, 1, 1, null);

        double x = (matDest.cols() - s.width) / 2;
        double y = rect.y + rect.height + s.height + 5;

        Imgproc.putText(matDest, msg, new Point(x, y), font, 1, TEXT_COLOR, 2);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemType = menu.add(detectorName[detectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.d(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemType)
            detectorName[detectorType] = "Java";

        return true;
    }

    private void setMinFaceSize(float faceSize) {
        relativeFaceSize = faceSize;
        absoluteFaceSize = 0;
    }
}
